import json
import os
import statistics
import sys
import time
import socket
import shutil
import torch
import yaml
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets import CaptionDataset, CaptionDataset_transformer

sys.path.append("./models")
# from models_i2u import ImageToUnit
from models_k import ImageToUnit


def train(device, loader, model, reconstruction_loss, optimizer):
    accuracies = []
    losses = []

    for imgs, units, seq_lens, padding_masks in loader:
        imgs = imgs.to(device)
        units = units.to(device)
        seq_lens = seq_lens.to(device)
        padding_masks = padding_masks.to(device)
        seq_lens = seq_lens.squeeze(1)

        logits, kl_loss = model(imgs, units, seq_lens, padding_masks)

        logits, _, _, _ = pack_padded_sequence(
            logits, (seq_lens - 1).cpu().tolist(), batch_first=True, enforce_sorted=False
        )
        targets, _, _, _ = pack_padded_sequence(
            units[:, 1:], (seq_lens - 1).cpu().tolist(), batch_first=True, enforce_sorted=False
        )

        loss = reconstruction_loss(logits, targets) + kl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = torch.sum(torch.argmax(logits, dim=1) == targets) / logits.size(0)
        accuracies.append(accuracy.item())
        losses.append(loss.item())

    return statistics.mean(accuracies), statistics.mean(losses)


@torch.inference_mode()
def validate(device, loader, model, reconstruction_loss):
    accuracies = []
    losses = []

    for imgs, units, seq_lens, padding_masks, _ , _ in loader:
        imgs = imgs.to(device)
        units = units.to(device)
        seq_lens = seq_lens.to(device)
        padding_masks = padding_masks.to(device)
        seq_lens = seq_lens.squeeze(1)

        logits, kl_loss = model(imgs, units, seq_lens, padding_masks)

        logits, _, _, _ = pack_padded_sequence(
            logits, (seq_lens - 1).cpu().tolist(), batch_first=True, enforce_sorted=False
        )
        targets, _, _, _ = pack_padded_sequence(
            units[:, 1:], (seq_lens - 1).cpu().tolist(), batch_first=True, enforce_sorted=False
        )

        loss = reconstruction_loss(logits, targets) + kl_loss

        accuracy = torch.sum(torch.argmax(logits, dim=1) == targets) / logits.size(0)
        accuracies.append(accuracy.item())
        losses.append(loss.item())

    return statistics.mean(accuracies), statistics.mean(losses)


def main(config):

    is_debug = True if sys.gettrace() else False
    if is_debug:
        print("Debugging Mode")
        train_ID = "debugging_ICASSP_model"
    else:
        print("Training mode")
        train_ID = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )[2:].replace(" ", "_") + "_ICASSP" + f"_{socket.gethostname()}"

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dir_name = config["I2U"]["data_folder"]
    if not os.path.isdir(f"../../saved_model/I2U/{dir_name}"):
        os.mkdir(f"../../saved_model/I2U/{dir_name}")
    if not os.path.isdir(f"../../saved_model/I2U/{dir_name}/{train_ID}"):
        os.mkdir(f"../../saved_model/I2U/{dir_name}/{train_ID}")

    shutil.copyfile("../../config_ICASSP.yml", f"../../saved_model/I2U/{dir_name}/{train_ID}/config_ICASSP.yml")

    if os.path.exists(f'../../data/processed/{dir_name}/'):
        data_folder = f'../../data/processed/{dir_name}/'  # folder with data files saved by create_input_files.py
    elif os.path.exists(f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}/'):
        data_folder = f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}/'
    else:
        raise ValueError(f"Dir: {dir_name} doesn't exist. Please check.")

    data_name = f'coco_{str(config["I2U"]["captions_per_image"])}_cap_per_img_{str(config["I2U"]["min_word_freq"])}_min_word_freq'  # base name shared by data files
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    
    # data_folder = os.path.join(os.path.dirname(__file__), "../..", config["I2U"]["data_folder"])
    # word_map_path = os.path.join(os.path.dirname(__file__), "../..", config["I2U"]["word_map"])
    # model_path = os.path.join(os.path.dirname(__file__), "../..", config["I2U"]["model_path"])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])

    train_loader = DataLoader(
        CaptionDataset_transformer(data_folder, data_name, "TRAIN", transform=transform),
        config["I2U"]["batch_size"],
        shuffle=True,
        num_workers=config["I2U"]["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        CaptionDataset_transformer(data_folder, data_name, "VAL", transform=transform),
        config["I2U"]["batch_size"],
        num_workers=config["I2U"]["num_workers"],
        pin_memory=True,
    )

    # with open(word_map_path) as j:
    #     word_map = json.load(j)

    model = ImageToUnit(word_map=word_map, max_len=config["I2U"]["max_len"]).to(device)
    reconstruction_loss = nn.CrossEntropyLoss(ignore_index=word_map["<pad>"], reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["I2U"]["lr"])

    if config["I2U"]["pretrained_model"] != "":
        ckpt = torch.load(config["I2U"]["pretrained_model"])
        ckpt_new = ckpt.copy()
        for k in ckpt.keys():
            if "image_encoder" in k:
                # print(k)
                ckpt_new["image_encoder." + k] = ckpt_new[k]
                del ckpt_new[k]
        model.load_state_dict(ckpt_new, strict=False)
        print(f"Loading checkpoint {config['I2U']['pretrained_model']}")


    max_accuracy = 0
    for epoch in range(1, 1 + config["I2U"]["epoch"]):
        train_accuracy, train_loss = train(device, train_loader, model, reconstruction_loss, optimizer)
        val_accuracy, val_loss = validate(device, val_loader, model, reconstruction_loss)

        print(
            "epoch",
            epoch,
            "train_accuracy",
            train_accuracy,
            "train_loss",
            train_loss,
            "val_accuracy",
            val_accuracy,
            "val_loss",
            val_loss,
            flush=True,
        )

        if max_accuracy < val_accuracy:
            max_accuracy = val_accuracy
            # f"../../saved_model/I2U/{dir_name}/{train_ID}/" + "i2u_with_sentence_embedding.pt"
            torch.save(model.state_dict(), f"../../saved_model/I2U/{dir_name}/{train_ID}/" + "i2u_with_sentence_embedding.pt")


if __name__ == "__main__":
    with open("../../config_ICASSP.yml") as y:
        config = yaml.safe_load(y)

    main(config)
