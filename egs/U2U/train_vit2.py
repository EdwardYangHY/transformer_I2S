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

print_freq = 100

def get_lr_schedule(optimizer, num_warmup_epochs: int = 10, d_model: int = 768):

    def lr_lambda(current_epoch: int):
        """
        Eq. (3) in [Transformer paper](https://arxiv.org/abs/1706.03762)
        """
        return d_model**(-0.5) * min((current_epoch+1)**(-0.5), (current_epoch+1)*num_warmup_epochs**(-1.5))

    return LambdaLR(optimizer, lr_lambda, verbose=True)


def train(model, reconstruct_loss, train_loader, optimizer, device, kl_weight, epoch = 0):
    losses = []
    accuracies = []
    for i, (units, padding_masks, seq_lens, imgs) in enumerate(train_loader):
        units = units.to(device)
        padding_masks = padding_masks.to(device)
        seq_lens = seq_lens.to(device)
        imgs = imgs.to(device)

        logits, kl_loss = model(units, padding_masks, seq_lens, imgs, use_encoder = True)
        
        # Remove padding
        logits, _, _, _ = pack_padded_sequence(logits, (seq_lens-1).cpu().tolist(), batch_first=True, enforce_sorted=False)
        targets, _, _, _ = pack_padded_sequence(units[:, 1:], (seq_lens-1).cpu().tolist(), batch_first=True, enforce_sorted=False)  # remove <start>

        loss = reconstruct_loss(logits, targets) + kl_weight*kl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = torch.sum(torch.argmax(logits, dim=1) == targets) / logits.size(dim=0)

        this_loss = loss.item()
        this_accuracy = accuracy.item()

        losses.append(loss.item())
        accuracies.append(accuracy.item())
        if i % print_freq == 0:
            #sys.stdout = Logger("../../model/U2U/transformer_vit1/vit1_log.txt")
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f} \t'
                  'Top-5 Accuracy {accuracies:.3f}'.format(epoch, i, len(train_loader),
                                                                            loss=this_loss,
                                                                            accuracies=this_accuracy))
    return statistics.mean(losses), statistics.mean(accuracies)


# def validate(model, loader, device, word_map):
#     num_successful_decode = 0
#     for units, padding_masks, seq_lens, imgs in loader:
#         units = units.to(device)
#         padding_masks = padding_masks.to(device)
#         imgs = imgs.to(device)

#         decoded_units = model.decode(word_map['<start>'], word_map['<end>'], x=units, padding_mask=padding_masks, image=imgs)

#         if seq_lens[0] == len(decoded_units) and torch.all(units[0, :seq_lens[0]] == torch.tensor(decoded_units, device=device)):
#             num_successful_decode += 1
#     return num_successful_decode / len(loader)

def validate(model, reconstruct_loss, val_loader, optimizer, device, kl_weight, epoch = 0):
    model.eval()
    losses = []
    accuracies = []
    with torch.no_grad():
        for i, (units, padding_masks, seq_lens, imgs) in enumerate(val_loader):
            units = units.to(device)
            padding_masks = padding_masks.to(device)
            seq_lens = seq_lens.to(device)
            imgs = imgs.to(device)

            logits, kl_loss = model(units, padding_masks, seq_lens, imgs, use_encoder = True)
            
            # Remove padding
            logits, _, _, _ = pack_padded_sequence(logits, (seq_lens-1).cpu().tolist(), batch_first=True, enforce_sorted=False)
            targets, _, _, _ = pack_padded_sequence(units[:, 1:], (seq_lens-1).cpu().tolist(), batch_first=True, enforce_sorted=False)  # remove <start>

            loss = reconstruct_loss(logits, targets) + kl_weight*kl_loss

            accuracy = torch.sum(torch.argmax(logits, dim=1) == targets) / logits.size(dim=0)

            this_loss = loss.item()
            this_accuracy = accuracy.item()
            losses.append(loss.item())
            accuracies.append(accuracy.item())
            if i % print_freq == 0:
                #sys.stdout = Logger("../../model/U2U/transformer_vit1/vit1_log.txt")
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss:.4f} \t'
                    'Top-5 Accuracy {accuracies:.3f}'.format(epoch, i, len(val_loader),
                                                                                loss=this_loss,
                                                                                accuracies=this_accuracy))
    return statistics.mean(losses), statistics.mean(accuracies)

def main(config):
    # Load word map
    with open(config["i2u"]["wordmap"]) as j:
        word_map = json.load(j)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(
        UnitImageDataset(config["u2u"]["data_folder"], config["u2u"]["data_name"], 'TRAIN'),
        batch_size=config["u2u"]["batch_size"],
        shuffle=True,
        num_workers=config["u2u"]["num_workers"],
        pin_memory=True,
        )
    val_loader = DataLoader(
        UnitImageDataset(config["u2u"]["data_folder"], config["u2u"]["data_name"], 'VAL'),
        batch_size=config["u2u"]["batch_size"],
        shuffle=False,
        num_workers=config["u2u"]["num_workers"],
        pin_memory=True,
        )
    
    model = TransformerVAEwithCNN(len(word_map), config["u2u"]["d_embed"]).to(device)
    reconstruct_loss = nn.CrossEntropyLoss(reduction="sum").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["u2u"]["lr"])
    scheduler = get_lr_schedule(optimizer, num_warmup_epochs=config["u2u"]["warmup_epoch2"], d_model=config["u2u"]["d_model"])

    max_val_accuracy = 0
    for epoch in range(config["u2u"]["epoch2"]):
        # val_loss, val_accuracy = validate(model, reconstruct_loss, val_loader, optimizer, device, kl_weight=config["u2u"]["kl_weight"], epoch = epoch)
        loss, accuracy = train(model, reconstruct_loss, train_loader, optimizer, device, kl_weight=config["u2u"]["kl_weight"], epoch = epoch)
        print("epoch", epoch, "loss", loss, "accuracy", accuracy, flush=True)
        val_loss, val_accuracy = validate(model, reconstruct_loss, val_loader, optimizer, device, kl_weight=config["u2u"]["kl_weight"], epoch = epoch)
        print("epoch", epoch, "val accuracy", val_accuracy, flush=True)
        if max_val_accuracy < val_accuracy:
            max_val_accuracy = val_accuracy
            torch.save(model.state_dict(), config["u2u"]["path2"])
        scheduler.step()



if __name__ == "__main__":
    with open('../../config.yml') as yml:
        config = yaml.safe_load(yml)
    
    main(config)