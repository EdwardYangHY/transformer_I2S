import json
import statistics

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import yaml

from models import TransformerVAEwithViT
from dataset import UnitImageDataset


def get_lr_schedule(optimizer, num_warmup_epochs: int = 10, d_model: int = 768):

    def lr_lambda(current_epoch: int):
        """
        Eq. (3) in [Transformer paper](https://arxiv.org/abs/1706.03762)
        """
        return d_model**(-0.5) * min((current_epoch+1)**(-0.5), (current_epoch+1)*num_warmup_epochs**(-1.5))

    return LambdaLR(optimizer, lr_lambda, verbose=True)


def train(model, reconstruct_loss, train_loader, optimizer, device, kl_weight):
    losses = []
    accuracies = []
    for units, padding_masks, seq_lens, imgs in train_loader:
        units = units.to(device)
        padding_masks = padding_masks.to(device)
        seq_lens = seq_lens.to(device)
        imgs = imgs.to(device)

        logits, kl_loss = model(units, padding_masks, seq_lens, imgs)
        
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


def validate(model, loader, device, word_map):
    num_successful_decode = 0
    for units, padding_masks, seq_lens, imgs in loader:
        units = units.to(device)
        padding_masks = padding_masks.to(device)
        imgs = imgs.to(device)

        decoded_units = model.decode(word_map['<start>'], word_map['<end>'], x=units, padding_mask=padding_masks, image=imgs)

        if seq_lens[0] == len(decoded_units) and torch.all(units[0, :seq_lens[0]] == torch.tensor(decoded_units, device=device)):
            num_successful_decode += 1
    return num_successful_decode / len(loader)


def main(config):
    # Load word map
    with open(config["i2u"]["wordmap"]) as j:
        word_map = json.load(j)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(
        UnitImageDataset(config["u2u"]["data_folder_incremental"], config["u2u"]["data_name"], 'TRAIN'),
        batch_size=config["u2u"]["batch_size"],
        shuffle=True,
        num_workers=config["u2u"]["num_workers"],
        pin_memory=True,
        )
    
    model = TransformerVAEwithViT(len(word_map), config["u2u"]["d_embed"]).to(device)
    model.load_state_dict(torch.load(config["u2u"]["path2"]))
    reconstruct_loss = nn.CrossEntropyLoss(reduction="sum").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["u2u"]["lr"])
    scheduler = get_lr_schedule(optimizer, num_warmup_epochs=config["u2u"]["warmup_epoch2_incremental"], d_model=config["u2u"]["d_model"])

    for epoch in range(config["u2u"]["epoch2_incremental"]):
        loss, accuracy = train(model, reconstruct_loss, train_loader, optimizer, device, kl_weight=config["u2u"]["kl_weight"])
        print("epoch", epoch, "loss", loss, "accuracy", accuracy, flush=True)
        scheduler.step()
    torch.save(model.state_dict(), config["u2u"]["path2_incremental"])


if __name__ == "__main__":
    with open('../../config.yml') as yml:
        config = yaml.safe_load(yml)
    
    main(config)