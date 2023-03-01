import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
# from models import TransformerLM, TransformerConditionedLM, TransformerSentenceLM
from models_modified import TransformerSentenceLM_FixedImg, TransformerSentenceLM_FixedImg_gated
from datasets import *
from transformer_I2S.egs.I2U.utils import *       #changed
from nltk.translate.bleu_score import corpus_bleu
# from torch.optim.lr_scheduler import LambdaLR
import shutil
import trainer

from torch.utils.tensorboard import SummaryWriter

import yaml

config_path = '../../config_sentence.yml'
with open(config_path, 'r') as yml:
    config = yaml.safe_load(yml)

dir_name = config["i2u"]["dir_name"]
model_params = config["i2u"]["model_params"]
train_params = config["i2u"]["train_params"]

# Data parameters
data_folder = f'../../data/processed/{dir_name}/'  # folder with data files saved by create_input_files.py
data_name = f'coco_{str(config["i2u"]["captions_per_image"])}_cap_per_img_{str(config["i2u"]["min_word_freq"])}_min_word_freq'  # base name shared by data files

LM_checkpoint = "/net/papilio/storage2/yhaoyuan/transformer_I2S/saved_model/LM/SpokenCOCO_LibriSpeech/PP_15.6512/checkpoint_coco_1_cap_per_img_1_min_word_freq.pth.tar"


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
checkpoint = train_params["checkpoint_path"]  # path to checkpoint, None if none
use_scheduler = train_params["use_scheduler"]

kl_weight = train_params["kl_weight"]

import sys
is_debug = True if sys.gettrace() else False
if is_debug:
    print("Debugging Mode")
    train_ID = "debugging_uLM_sentence"
else:
    print("Training mode")
    train_ID = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )[2:].replace(" ", "_") + "_uLM_sentence"

def main():
    """
    Training and validation.
    """

    global best_bleu4, best_accuracy, checkpoint, start_epoch, data_name, word_map, device

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    model_params['vocab_size'] = len(word_map)
    # ----------------------------------------------------------------
    img_refine_params = config["i2u"]["refine_encoder_params"]
    model_params["refine_encoder_params"] = img_refine_params
    # ----------------------------------------------------------------

    ### Not gated model ###
    if train_params["gated_decoder"]:
        model = TransformerSentenceLM_FixedImg_gated(**model_params)
    else:
        model = TransformerSentenceLM_FixedImg(**model_params)
    
    if train_params["load_uLM"]:
        model.load_Pretrained_LM(LM_checkpoint)

    optimizer = getattr(torch.optim, train_params["optimizer"])(model.parameters(), lr=train_params["lr"])

    # scheduel LR
    if use_scheduler:
        scheduler = get_lr_schedule(optimizer, train_params["warmup_epoch"], model_params["d_model"])
    if checkpoint is not None:
        model, optimizer, start_epoch, best_bleu4, best_accuracy = load_checkpoint(checkpoint, model, optimizer, device)
    
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
    
    val_loader_beam = torch.utils.data.DataLoader(
        CaptionDataset_transformer(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=workers, pin_memory=True)
    
    # Epochs
    writer = SummaryWriter(f"../../saved_model/I2U/{dir_name}/{train_ID}/log")
    # writer = None
    # Copy config to present model dir to keep record
    shutil.copyfile(config_path, f"../../saved_model/I2U/{dir_name}/{train_ID}/config_sentence.yml")

    for epoch in range(start_epoch, epochs):
        trainer.train(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            writer=writer,
            grad_clip=grad_clip,
            device=device,
            print_freq=print_freq,
            kl_weight=kl_weight
        )
        if use_scheduler:
            writer.add_scalar("Train/lr", float(scheduler.get_last_lr()[-1]), epoch)
            scheduler.step()

        recent_bleu4 = trainer.validate_beam(
            val_loader=val_loader_beam,
            model=model,
            start_unit=word_map["<start>"],
            end_unit=word_map["<end>"],
            epoch=epoch,
            writer=writer,
            decode_num=10,
            device=device
        )

        # trainer.validate(
        #     val_loader=val_loader,
        #     model=model,
        #     criterion=criterion,
        #     epoch=epoch,
        #     writer=writer,
        #     device=device,
        #     start_unit=word_map["<start>"],
        #     end_unit=word_map["<end>"],
        #     print_freq=print_freq,
        #     kl_weight=kl_weight
        # )

        print(f"Beam Search Validation: {recent_bleu4}")

        is_best_bleu4 = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        start = time.time()

        save_checkpoint(
            data_name=data_name,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            metric_name="bleu-4",
            metric_value=recent_bleu4,
            is_best=is_best_bleu4,
            train_ID=train_ID,
            device=device
        )
        # save_checkpoint("bleu-4", data_name, epoch, model, optimizer, recent_bleu4, 0, is_best_bleu4, train_ID, device)
        print(f"Saving model in {time.time() - start} seconds")
        
if __name__ == '__main__':
    main()

# def train(train_loader, model, criterion, optimizer, epoch, writer):
#     """
#     Performs one epoch's training.

#     :param train_loader: DataLoader for training data
#     :param encoder: encoder model
#     :param decoder: decoder model
#     :param criterion: loss layer
#     :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
#     :param decoder_optimizer: optimizer to update decoder's weights
#     :param epoch: epoch number
#     """

#     model.train()

#     batch_time = AverageMeter()  # forward prop. + back prop. time
#     data_time = AverageMeter()  # data loading time
#     losses = AverageMeter()  # loss (per word decoded)
#     top5accs = AverageMeter()  # top5 accuracy

#     start = time.time()

#     # Batches
#     for i, (imgs, caps, caplens, padding_mask) in enumerate(train_loader):
#         data_time.update(time.time() - start)

#         # Move to GPU, if available
#         imgs = imgs.to(device)
#         caps = caps.to(device)
#         caplens = caplens.to(device)
#         caplens = caplens.squeeze()
#         padding_mask = padding_mask.to(device)

#         # Forward prop.
#         logits, encoded_seq, decode_lengths, sort_ind, kl_loss = model(
#             imgs,
#             caps,
#             padding_mask,
#             caplens
#         )
#         # x = model.embed(caps)
#         # x = model.pos_encoder(x)
#         # z = model.sentence_encoder(x, src_key_padding_mask = padding_mask)
#         # z = z * padding_mask.logical_not().unsqueeze(2)
#         # z = z.sum(dim = 1)/ caplens.unsqueeze(1)
#         # mu = model.mu(z)  # (batch, sentence_embed)
#         # imgs, gx = model.image_encoder(imgs[0].unsqueeze(0))
#         # action = imgs.flatten()
#         # action = torch.cat([action, mu[0].flatten()]).unsqueeze(0)
#         # seq = model.decode(action=action, start_unit=word_map["<start>"], end_unit=word_map["<end>"], max_len=130, beam_size=10)

#         # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
#         targets = encoded_seq[:, 1:]

#         # Remove timesteps that we didn't decode at, or are pads
#         # pack_padded_sequence is an easy trick to do this
#         scores, _, _, _ = pack_padded_sequence(logits, decode_lengths, batch_first=True, enforce_sorted=False)
#         targets, _, _, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=False)

#         # Calculate loss
#         loss = criterion(scores, targets) + kl_weight*kl_loss

#         # Back prop.
#         optimizer.zero_grad()
#         loss.backward()

#         # Grad Clip preveting grad explosion
#         if grad_clip is not None:
#             clip_gradient(optimizer, grad_clip)

#         optimizer.step()
#         # this_accuracy = torch.sum(torch.argmax(scores, dim=1) == targets) / logits.size(dim=0)
        
#         # Keep tracks of metrics
#         top5 = accuracy(scores, targets, 5)
#         losses.update(loss.item(), sum(decode_lengths))
#         top5accs.update(top5, sum(decode_lengths))
#         batch_time.update(time.time() - start)

#         if i % 10 == 0:
#             niter = epoch * len(train_loader) + i
#             writer.add_scalar('Train/Loss', loss.data, niter)
#             writer.add_scalar('Train/Top-5-Accuracy', top5, niter)

#         start = time.time()

#         if i % print_freq == 0:
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
#                                                                           batch_time=batch_time,
#                                                                           data_time=data_time, loss=losses,
#                                                                           top5=top5accs))

# def validate_beam(val_loader, model, criterion, num = 100):
#     """
#         Bleu-4 Score should not be computed by decoded seqs from GT ans.
#         It should compute between generated sequence (by beam-search or other generative decoding)
#         and the GT seqs.
#     """
#     model.eval()
#     # batch_time = AverageMeter()
#     # We don't have loss or accuracy if we use beam search to decode.
#     # losses = AverageMeter()
#     # top5accs = AverageMeter()
#     # bleu_4 = AverageMeter()
#     start = time.time()

#     start_unit = word_map["<start>"]
#     end_unit = word_map["<end>"]

#     refs = list() # GT captions
#     hypos = list() # pred captions
#     zero_count = 0
#     for i, (imgs, caps, caplens, padding_mask, all_caps, all_padding_mask) in enumerate(iter(val_loader)):
#         imgs = imgs.to(device)
#         caps = caps.to(device)
#         caplens = caplens.to(device)
#         caplens = caplens.squeeze()
#         padding_mask = padding_mask.to(device)
#         all_caps = all_caps.to(device)
#         all_padding_mask = all_padding_mask.to(device)

#         pred_seq = model.decode(
#             start_unit = start_unit,
#             end_unit = end_unit,
#             image = imgs,
#             max_len = 150,
#             beam_size = 5,
#         )

#         pred_seq = [w for w in pred_seq if w not in {word_map['<start>'], word_map['<pad>']}]
#         if len(pred_seq) == 0:
#             zero_count += 1
#         if i == 5 and zero_count >= 3:
#             return 0

#         hypos.append(pred_seq)
#         # hypos

#         for j in range(all_caps.shape[0]):
#             img_caps = all_caps[j].tolist()
#             img_captions = list(
#                 map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
#                     img_caps))  # remove <start> and pads
#             refs.append(img_captions)
#         assert len(hypos) == len(refs)
        
#         if i >= num:
#             break

#     bleu_4 = corpus_bleu(refs, hypos)
#     print(f"Valid Time: {time.time() - start}")
#     return bleu_4

# def validate(val_loader, model, criterion): # -> bleu-4, accuracy
#     model.eval()
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     top5accs = AverageMeter()
#     start = time.time()

#     refs = list() # GT captions
#     hypos = list() # pred captions

#     with torch.no_grad():
#         for i, (imgs, caps, caplens, padding_mask, all_caps, all_padding_mask) in enumerate(val_loader):
#             imgs = imgs.to(device)
#             caps = caps.to(device)
#             caplens = caplens.to(device)
#             caplens = caplens.squeeze()
#             padding_mask = padding_mask.to(device)
#             all_caps = all_caps.to(device)
#             all_padding_mask = all_padding_mask.to(device)

#             logits, encoded_seq, decode_lengths, sort_ind, kl_loss = model(
#                 imgs,
#                 caps,
#                 padding_mask,
#                 caplens
#             )
#             targets = encoded_seq[:, 1:]
#             scores_copy = logits.clone()
#             scores, _, _, _ = pack_padded_sequence(logits, decode_lengths, batch_first=True, enforce_sorted=False)
#             targets, _, _, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=False)
            
#             loss = criterion(scores, targets) + kl_weight*kl_loss

#             # Keep track of current validations
#             losses.update(loss.item(), sum(decode_lengths))
#             top5 = accuracy(scores, targets, 5)
#             top5accs.update(top5, sum(decode_lengths))
#             batch_time.update(time.time() - start)
            
#             # niter = epoch * len(train_loader) + i
#             # writer.add_scalar('Train/Loss', loss.data, niter)
#             # writer.add_scalar('Train/Top-5-Accuracy', top5, niter)

#             start = time.time()

#             if i % print_freq == 0:
#                 print('Validation: [{0}/{1}]\t'
#                       'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
#                                                                                 loss=losses, top5=top5accs))

#             # Store references (true captions), and hypothesis (prediction) for each image
#             # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
#             # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

#             # References
#             all_caps = all_caps[sort_ind]  # because images were sorted in the decoder
#             for j in range(all_caps.shape[0]):
#                 img_caps = all_caps[j].tolist()
#                 img_captions = list(
#                     map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
#                         img_caps))  # remove <start> and pads
#                 refs.append(img_captions)

#             # Hypotheses
#             _, preds = torch.max(scores_copy, dim=2)
#             preds = preds.tolist()
#             temp_preds = list()
#             for j, p in enumerate(preds):
#                 temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
#             preds = temp_preds
#             hypos.extend(preds)

#             assert len(refs) == len(hypos)
#         # Calculate BLEU-4 scores
#         bleu4 = corpus_bleu(refs, hypos)

#         # Write to tensorboard
#         # niter = epoch
#         # writer.add_scalar('Valid/Loss', losses.avg, niter)
#         # writer.add_scalar('Valid/Top-5-Accuracy', top5accs.avg, niter)
#         # writer.add_scalar('Valid/BLEU-4', bleu4, niter)

#         print(
#             '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
#                 loss=losses,
#                 top5=top5accs,
#                 bleu=bleu4))
#     return bleu4, top5accs.avg, losses.avg


