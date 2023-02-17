import json
import os

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
import yaml

from models import TransformerVAE, TransformerAE
from dataset import UnitDataset


def train(model, reconstruct_loss, train_loader, optimizer, device, kl_weight):
    losses = []
    accuracies = []
    for units, padding_masks, seq_lens in train_loader:
        units = units.to(device)
        padding_masks = padding_masks.to(device)
        seq_lens = (seq_lens-1).tolist()

        logits, kl_loss = model(units, padding_masks)
        
        # Remove padding
        logits, _, _, _ = pack_padded_sequence(logits, seq_lens, batch_first=True, enforce_sorted=False)
        targets, _, _, _ = pack_padded_sequence(units[:, 1:], seq_lens, batch_first=True, enforce_sorted=False)  # remove <start>

        loss = reconstruct_loss(logits, targets) + kl_weight*kl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = torch.sum(torch.argmax(logits, dim=1) == targets) / logits.size(dim=0)

        losses.append(loss.item())
        accuracies.append(accuracy.item())
    return losses, accuracies


def main(config, word_map):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(
        UnitDataset(data_folder, data_name, 'TRAIN'),
        batch_size=config["u2u"]["batch_size"],
        shuffle=True,
        num_workers=config["u2u"]["num_workers"],
        pin_memory=True,
        )
    
    if config["u2u"]["model"] == "ae":
        model = TransformerAE(len(word_map), config["u2u"]["d_embed"]).to(device)
    elif config["u2u"]["model"] == "vae":
        model = TransformerVAE(len(word_map), config["u2u"]["d_embed"]).to(device)
    else:
        raise ValueError("Not supported model")
    reconstruct_loss = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["u2u"]["lr"])

    losses = []
    accuracies = []
    for _ in range(config["u2u"]["epoch"]):
        loss, accuracy = train(model, reconstruct_loss, train_loader, optimizer, device, kl_weight=config["u2u"]["kl_weight"])
        losses += loss
        accuracies += accuracy
    
    print("loss:", losses)
    print("accuracy:", accuracies)
    torch.save(model.state_dict(), config["u2u"]["path"])


if __name__ == "__main__":
    with open('../../config.yml') as yml:
        config = yaml.safe_load(yml)

    # Data parameters
    data_folder = '../../data/I2U/processed/'  # folder with data files saved by create_input_files.py
    data_name = f'coco_{str(config["i2u"]["captions_per_image"])}_cap_per_img_{str(config["i2u"]["min_word_freq"])}_min_word_freq'  # base name shared by data files

    # Load word map
    with open(os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')) as j:
        word_map = json.load(j)
    
    main(config, word_map)