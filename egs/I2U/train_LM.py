import time
import sys
import yaml
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from datasets import *
from utils_i2u import *
import shutil
import trainer
from glob import glob

from torch.utils.tensorboard import SummaryWriter

sys.path.append("./models")
from models import TransformerLM, TransformerConditionedLM, TransformerSentenceLM

config_path = "../../config_LM.yml"
with open(config_path, 'r') as yml:
    config = yaml.safe_load(yml)

dir_name = config["i2u"]["dir_name"]
model_params = config["i2u"]["model_params"]
train_params = config["i2u"]["train_params"]

# Data parameters
# data_folder = '/media/ssd/caption data'  # folder with data files saved by create_input_files.py
data_folder = f'../../data/processed/{dir_name}/'  # folder with data files saved by create_input_files.py
# data_name = 'coco_4_cap_per_img_5_min_word_freq'  # base name shared by data files
#data_name = f'coco_{str(config["i2u"]["captions_per_image"])}_cap_per_img_{str(config["i2u"]["min_word_freq"])}_min_word_freq'  # base name shared by data files
data_name = f'coco_{str(config["i2u"]["captions_per_image"])}_cap_per_img_{str(config["i2u"]["min_word_freq"])}_min_word_freq'  # base name shared by data files

# Model parameters
# emb_dim = 512  # dimension of word embeddings
# attention_dim = 512  # dimension of attention linear layers
# decoder_dim = 512  # dimension of decoder RNN
# dropout = 0.5
# import torch
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.cuda.is_available()) #查看cuda是否可用)
# print(torch.backends.cudnn.version())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
# device = torch.device("cpu")
print("Training on device {}".format(device.type))
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = train_params["epoch"]  # number of epochs to train for (if early stopping is not triggered)
batch_size = train_params["batch_size"]
workers = train_params["num_workers"]  # for data-loading; right now, only 1 works with h5py
# encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
# decoder_lr = 4e-4  # learning rate for decoder
grad_clip = train_params["grad_clip"]  # clip gradients at an absolute value of
best_bleu4 = 0.  # BLEU-4 score right now
best_accuracy = 0.
best_perplexity = 100000
print_freq = train_params["print_freq"]  # print training/validation stats every __ batches
# fine_tune_encoder = model_params["fine_tune_image_encoder"]  # fine-tune encoder?
checkpoint = train_params["checkpoint_path"]  # path to checkpoint, None if none
#checkpoint = "/net/papilio/storage2/yhaoyuan/LAbyLM/model/I2U/gtts_3_captions/BEST_checkpoint_coco_3_cap_per_img_1_min_word_freq_gpu.pth.tar"
use_scheduler = train_params["use_scheduler"]

kl_weight = train_params["kl_weight"]

import sys
is_debug = True if sys.gettrace() else False
if is_debug:
    print("Debugging Mode")
    train_ID = "debugging_sentence"
else:
    print("Training mode")
    train_ID = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )[2:].replace(" ", "_") + "_sentence"
# train_ID = "debugging"

# def get_lr_schedule(optimizer, num_warmup_epochs: int = 10, d_model: int = 2048):
#     def lr_lambda(current_epoch: int):
#         """
#         Eq. (3) in [Transformer paper](https://arxiv.org/abs/1706.03762)
#         """
#         return d_model**(-0.5) * min((current_epoch+1)**(-0.5), (current_epoch+1)*num_warmup_epochs**(-1.5))

#     return LambdaLR(optimizer, lr_lambda, verbose=True)

def main():
    """
    Training and validation.
    """

    global best_bleu4, best_accuracy, best_perplexity, checkpoint, start_epoch, data_name, word_map, device

    # Read word map
    # word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    word_map_file = glob(data_folder+"/WORDMAP*.json")[0]
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    model_params['vocab_size'] = len(word_map)

    model = TransformerLM(**model_params)

    # optimizer = torch.optim.Adam(model.parameters(), train_params["lr"])
    optimizer = getattr(torch.optim, train_params["optimizer"])(model.parameters(), lr=train_params["lr"])

    # scheduel LR
    # if use_scheduler:
    #     scheduler = get_lr_schedule(optimizer, train_params["warmup_epoch"], model_params["d_model"])
    if checkpoint is not None:
        # model, optimizer, start_epoch, best_bleu4, best_accuracy, best_perplexity = load_checkpoint(checkpoint, model, optimizer, device)
        cp_state_dict = torch.load(checkpoint)
        cp_state_dict = cp_state_dict["model_state_dict"]
        model.load_state_dict(cp_state_dict, strict=True)

    if use_scheduler:
        scheduler = get_lr_schedule(optimizer, train_params["warmup_epoch"], last_epoch=start_epoch-1, d_model=model_params["d_model"])
        # scheduler = get_lr_schedule(optimizer, train_params["warmup_epoch"], d_model=model_params["d_model"])

    # set schedulaer's epoch to match up with the current lr
    # scheduler.last_epoch = start_epoch
    #optimizer = torch.optim.Adam(model.parameters(), decoder_lr)
    # Move to GPU, if available
    model.to(device)
    

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    train_loader = torch.utils.data.DataLoader(
        UnitDatasetMask(data_folder, data_name, 'TRAIN', word_map=word_map),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        UnitDatasetMask(data_folder, data_name, 'VAL', word_map=word_map),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    
    # Epochs
    writer = SummaryWriter(f"../../saved_model/LM/{dir_name}/{train_ID}/log")

    # Copy config to present model dir to keep record
    shutil.copyfile(config_path, f"../../saved_model/LM/{dir_name}/{train_ID}/config_LM.yml")

    for epoch in range(start_epoch, epochs):
        trainer.train_LM(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            writer=writer,
            grad_clip=grad_clip,
            device=device,
            print_freq=print_freq,
        )
        if use_scheduler:
            writer.add_scalar("Train/lr", float(scheduler.get_last_lr()[-1]), epoch)
            scheduler.step()

        pp, loss = trainer.validate_LM(
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            epoch=epoch,
            writer=writer,
            device=device,
            print_freq=print_freq,
        )


        is_best_perplexity = pp < best_perplexity
        best_perplexity = min(pp, best_perplexity)
        start = time.time()

        save_checkpoint_LM(
            data_name=data_name,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            metric_name="perplexity",
            metric_value=pp,
            is_best=is_best_perplexity,
            dir_name=dir_name,
            train_ID=train_ID,
            device=device
        )
        print(f"Saving model in {time.time() - start} seconds")


if __name__ == '__main__':
    main()
