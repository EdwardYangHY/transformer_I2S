import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import TransformerLM, TransformerConditionedLM
from datasets import *
from transformer_I2S.egs.I2U.utils_previous import *       #changed
from nltk.translate.bleu_score import corpus_bleu
# from torch.optim.lr_scheduler import LambdaLR
import shutil
import optuna
from optuna.trial import TrialState
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import yaml
with open('../../config.yml', 'r') as yml:
    config = yaml.safe_load(yml)

dir_name = config["i2u"]["dir_name"]
model_params = config["i2u"]["model_params"]
train_params = config["i2u"]["train_params"]


data_folder = f'../../data/processed/{dir_name}/'  # folder with data files saved by create_input_files.py
data_name = f'coco_{str(config["i2u"]["captions_per_image"])}_cap_per_img_{str(config["i2u"]["min_word_freq"])}_min_word_freq'  # base name shared by data files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
# device = torch.device("cpu")
print("Training on device {}".format(device.type))
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = train_params["epoch"]  # number of epochs to train for (if early stopping is not triggered)
batch_size = train_params["batch_size"]
workers = train_params["num_workers"]  # for data-loading; right now, only 1 works with h5py
grad_clip = train_params["grad_clip"]  # clip gradients at an absolute value of
best_bleu4 = 0.  # BLEU-4 score right now
best_accuracy = 0.
print_freq = train_params["print_freq"]  # print training/validation stats every __ batches
# fine_tune_encoder = model_params["fine_tune_image_encoder"]  # fine-tune encoder?
checkpoint = train_params["checkpoint_path"]  # path to checkpoint, None if none
# checkpoint = "/net/papilio/storage2/yhaoyuan/LAbyLM/model/I2U/gtts_3_captions/BEST_checkpoint_coco_3_cap_per_img_1_min_word_freq_gpu.pth.tar"
use_scheduler = train_params["use_scheduler"]

import sys
is_debug = True if sys.gettrace() else False
if is_debug:
    print("Debugging Mode")
    train_ID = "debugging"
else:
    print("Training mode")
    train_ID = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )[2:].replace(" ", "_")
# train_ID = "debugging"

def optimize_params(trial):
    model_params_trial = {
        # 'd_model': trial.suggest_categorical('d_model', [1024, 2048]),
        'layer_norm_eps': trial.suggest_float('layer_norm_eps', 1.0e-7, 1.0e-3, log=True),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5, step = 0.1),
        'image_backbone': trial.suggest_categorical('image_backbone', ["ResNet", "ViT"]),
        'use_refine_encoder': trial.suggest_categorical('use_refine_encoder', ["True", "False"]),
        'use_global_feature': trial.suggest_categorical('use_global_feature', ["True", "False"])
    }
    train_params_trial = {
        'lr': trial.suggest_float('lr', 1.0e-4, 1.0e-1, log=True),
        'optimizer': trial.suggest_categorical('optimizer', ["Adam", "AdamW"])
    }
    return model_params_trial, train_params_trial

def objective(trial):
    # Override params
    model_params_trial, train_params_trial = optimize_params(trial)
    for key, value in model_params_trial.items():
        model_params[key] = value
    for key, value in train_params_trial.items():
        train_params[key] = value
    
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    model_params["vocab_size"] = len(word_map)
    model = TransformerConditionedLM(**model_params)
    optimizer = getattr(torch.optim, train_params["optimizer"])(model.parameters(), lr=train_params["lr"])
    if use_scheduler:
        scheduler = get_lr_schedule(optimizer, train_params["warmup_epoch"], model_params["d_model"])
    # if checkpoint is not None:
    #     model, optimizer, start_epoch, best_bleu4, best_accuracy = load_checkpoint(checkpoint, model, optimizer, device)
    # Move to GPU, if available
    model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset_transformer(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset_transformer(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    
    if not os.path.isdir(f"../../saved_model/I2U/{dir_name}/optuna"):
        os.mkdir(f"../../saved_model/I2U/{dir_name}/optuna")
    shutil.copyfile("../../config.yml", f"../../saved_model/I2U/{dir_name}/optuna/config.yml")
    for epoch in tqdm(range(start_epoch, epochs)):
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            writer = None,
        )
        if use_scheduler:
            scheduler.step()
        # validata
        recent_bleu4, top5acc_avg, losses_avg = validate(
            val_loader, 
            model, 
            criterion
        )

        trial.report(top5acc_avg, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return top5acc_avg

def train(train_loader, model, criterion, optimizer, epoch, writer):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    model.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens, padding_mask) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        caplens = caplens.squeeze()
        padding_mask = padding_mask.to(device)

        # Forward prop.
        logits, encoded_seq, decode_lengths, sort_ind = model(
            imgs,
            caps,
            padding_mask,
            caplens
        )

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = encoded_seq[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _, _, _ = pack_padded_sequence(logits, decode_lengths, batch_first=True, enforce_sorted=False)
        targets, _, _, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=False)

        # Calculate loss
        loss = criterion(scores, targets)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Grad Clip preveting grad explosion
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        optimizer.step()

def validate(val_loader, model, criterion): # -> bleu-4, accuracy
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()
    start = time.time()

    refs = list() # GT captions
    hypos = list() # pred captions

    with torch.no_grad():
        for i, (imgs, caps, caplens, padding_mask, all_caps, all_padding_mask) in enumerate(val_loader):
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            caplens = caplens.squeeze()
            padding_mask = padding_mask.to(device)
            all_caps = all_caps.to(device)
            all_padding_mask = all_padding_mask.to(device)

            logits, encoded_seq, decode_lengths, sort_ind = model(
                imgs,
                caps,
                padding_mask,
                caplens
            )
            targets = encoded_seq[:, 1:]
            scores_copy = logits.clone()
            scores, _, _, _ = pack_padded_sequence(logits, decode_lengths, batch_first=True, enforce_sorted=False)
            targets, _, _, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=False)
            
            loss = criterion(scores, targets)

            # Keep track of current validations
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)
            
            # niter = epoch * len(train_loader) + i
            # writer.add_scalar('Train/Loss', loss.data, niter)
            # writer.add_scalar('Train/Top-5-Accuracy', top5, niter)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            all_caps = all_caps[sort_ind]  # because images were sorted in the decoder
            for j in range(all_caps.shape[0]):
                img_caps = all_caps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                refs.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypos.extend(preds)

            assert len(refs) == len(hypos)
        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(refs, hypos)

        # Write to tensorboard
        # niter = epoch
        # writer.add_scalar('Valid/Loss', losses.avg, niter)
        # writer.add_scalar('Valid/Top-5-Accuracy', top5accs.avg, niter)
        # writer.add_scalar('Valid/BLEU-4', bleu4, niter)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))
    return bleu4, top5accs.avg, losses.avg


if __name__ == '__main__':
    study = optuna.create_study(
        direction='maximize',
        storage="sqlite:///debug.sqlite3",  # Specify the storage URL here.
        study_name="optuna_debug",
        load_if_exists=True
    )
    study.optimize(objective, n_trials = 100)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    # main()
