import time
import yaml
import sys
import socket
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from datasets import *
from utils_i2u import *       #changed
import shutil
import trainer

from torch.utils.tensorboard import SummaryWriter


sys.path.append("./models")
# from models import models_modified
from models_modified import TransformerSentenceLM_FixedImg_Pool, TransformerSentenceLM_FixedImg_gated

# config_path = '../../config_sentence.yml'
config_path = '../../config_codec.yml'
with open(config_path, 'r') as yml:
    config = yaml.safe_load(yml)

print(f"Training data name: {config['i2u']['dir_name']}")
print(f"Training params: {config['i2u']['train_params']}")
print(f"Model params: {config['i2u']['model_params']}")

dir_name = config["i2u"]["dir_name"]
model_params = config["i2u"]["model_params"]
train_params = config["i2u"]["train_params"]

# Data parameters
# data_folder = f'../../data/processed/{dir_name}/'  # folder with data files saved by create_input_files.py
if os.path.exists(f'../../data/processed/{dir_name}/'):
    data_folder = f'../../data/processed/{dir_name}/'  # folder with data files saved by create_input_files.py
elif os.path.exists(f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}/'):
    data_folder = f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}/'
else:
    raise ValueError(f"Dir: {dir_name} doesn't exist. Please check.")

data_name = f'coco_{str(config["i2u"]["captions_per_image"])}_cap_per_img_{str(config["i2u"]["min_word_freq"])}_min_word_freq'  # base name shared by data files

# LM_checkpoint = "/net/papilio/storage2/yhaoyuan/transformer_I2S/saved_model/LM/SpokenCOCO_LibriSpeech/PP_15.6512/checkpoint_coco_1_cap_per_img_1_min_word_freq.pth.tar"
LM_checkpoint = "/net/papilio/storage2/yhaoyuan/transformer_I2S/saved_model/LM/Libri_Light_small_hubert_256/perplexity_6/perplexity_BEST_checkpoint_coco_1_cap_per_img_1_min_word_freq_gpu.pth.tar"

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
    train_ID = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )[2:].replace(" ", "_") + "_uLM_sentence" + f"_{socket.gethostname()}"

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
        model = TransformerSentenceLM_FixedImg_Pool(**model_params)
    
    if train_params["load_uLM"]:
        model.load_Pretrained_LM(LM_checkpoint)
        if train_params["freeze_uLM"] and not train_params["gated_decoder"]:
            model.freeze_LM()


    optimizer = getattr(torch.optim, train_params["optimizer"])(model.parameters(), lr=train_params["lr"])

    if checkpoint is not None:
        model, optimizer, start_epoch, best_bleu4, best_accuracy = load_checkpoint(
            checkpoint, 
            model, 
            optimizer, 
            device
        )

    # scheduel LR
    if use_scheduler:
        scheduler = get_lr_schedule(
            optimizer=optimizer, 
            num_warmup_epochs=train_params["warmup_epoch"], 
            d_model=model_params["d_model"],
            last_epoch=start_epoch-1
        )


    
    model.to(device)
    

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transform = transforms.Compose([normalize])
    # transform = None
    
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset_transformer(data_folder, data_name, 'TRAIN', transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset_transformer(data_folder, data_name, 'VAL', transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    
    val_loader_beam = torch.utils.data.DataLoader(
        CaptionDataset_transformer(data_folder, data_name, 'VAL', transform=transform),
        batch_size=1, shuffle=True, num_workers=workers, pin_memory=True)
    
    # Epochs
    writer = SummaryWriter(f"../../saved_model/I2U/{dir_name}/{train_ID}/log")
    # writer = None
    # Copy config to present model dir to keep record
    shutil.copyfile(config_path, f"../../saved_model/I2U/{dir_name}/{train_ID}/config_codec.yml")

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

        """
            可以考虑每 10 个epoch 做一次完整的 beam search validation？
            正常就用目前的validation就行 只是加一个罢了
        """
        recent_bleu4, top5acc, loss = trainer.validate(
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            epoch=epoch,
            writer=writer,
            device=device,
            start_unit=word_map["<start>"],
            end_unit=word_map["<end>"],
            print_freq=print_freq,
            kl_weight=kl_weight
        )

        # if epoch%10 == 0:
        #     recent_bleu4_beam = trainer.validate_beam(
        #         val_loader=val_loader_beam,
        #         model=model,
        #         start_unit=word_map["<start>"],
        #         end_unit=word_map["<end>"],
        #         epoch=epoch,
        #         writer=writer,
        #         decode_num=600,
        #         device=device
        #     )
        #     print(f"Beam Search Validation: {recent_bleu4_beam}")

        

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
            dir_name=dir_name,
            train_ID=train_ID,
            device=device
        )
        # save_checkpoint("bleu-4", data_name, epoch, model, optimizer, recent_bleu4, 0, is_best_bleu4, train_ID, device)
        print(f"Saving model in {time.time() - start} seconds")
        
if __name__ == '__main__':
    main()
