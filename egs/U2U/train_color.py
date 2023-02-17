import json
import statistics

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import yaml

from models import TransformerVAEwithCNN
from dataset import UnitImageDataset


def train(model, reconstruct_loss, train_loader, optimizer, device, kl_weight):
    losses = []
    accuracies = []
    for units, padding_masks, seq_lens, imgs in train_loader:
        units = units.to(device)
        padding_masks = padding_masks.to(device)
        seq_lens = seq_lens.to(device)
        imgs = imgs.to(device)

        logits, kl_loss = model(units, padding_masks, seq_lens, imgs, use_encoder=True)
        
        # Remove padding
        logits, _, _, _ = pack_padded_sequence(logits, (seq_lens-1).cpu().tolist(), batch_first=True, enforce_sorted=False)
        targets, _, _, _ = pack_padded_sequence(units[:, 1:], (seq_lens-1).cpu().tolist(), batch_first=True, enforce_sorted=False)  # remove <start>

        loss = reconstruct_loss(logits, targets) + kl_weight*kl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = torch.sum(torch.argmax(logits, dim=1) == targets) / logits.size(dim=0)

        losses.append(loss.item())
        accuracies.append(accuracy.item())
    return statistics.mean(losses), statistics.mean(accuracies)


def main(config):
    # Load word map
    with open(config["i2u"]["wordmap"]) as j:
        word_map = json.load(j)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(
        UnitImageDataset(config["u2u"]["data_folder"], config["u2u"]["data_name"], 'TRAIN'),
        batch_size=32,
        shuffle=True,
        num_workers=config["u2u"]["num_workers"],
        pin_memory=True,
        )
    
    model = TransformerVAEwithCNN(len(word_map), config["u2u"]["d_embed"]).to(device)
    reconstruct_loss = nn.CrossEntropyLoss(reduction="sum").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(30):
        loss, accuracy = train(model, reconstruct_loss, train_loader, optimizer, device, kl_weight=config["u2u"]["kl_weight"])
        print("epoch", epoch, "loss", loss, "accuracy", accuracy, flush=True)
        if epoch % 5 == 4:
            torch.save(model.state_dict(), f"../../model/U2U/transformer_cnn_{epoch}.pt")


if __name__ == "__main__":
    with open('../../config.yml') as yml:
        config = yaml.safe_load(yml)
    
    main(config)