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

import sys
class Logger(object):
    def __init__(self, file_path = "Default.log"):
        self.terminal = sys.stdout
        self.log = open(file_path, "a", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

from torch.utils.tensorboard import SummaryWriter

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

        logits, kl_loss = model(units, padding_masks, seq_lens, imgs, use_encoder=False)
        
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

            logits, kl_loss = model(units, padding_masks, seq_lens, imgs, use_encoder=False)
            
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


# def validate(model, device, word_map):
#     ans = [[] for _ in range(20)]

#     train_loader = DataLoader(
#         UnitImageDataset(config["u2u"]["data_folder"], config["u2u"]["data_name"], 'TRAIN'),
#         batch_size=1,
#         shuffle=False,
#         num_workers=config["u2u"]["num_workers"],
#         pin_memory=True,
#         )
    
#     val_loader = DataLoader(
#         UnitImageDataset(config["u2u"]["data_folder"], config["u2u"]["data_name"], 'VAL'),
#         batch_size=1,
#         shuffle=False,
#         num_workers=config["u2u"]["num_workers"],
#         pin_memory=True,
#         )

#     for i, (units, padding_masks, seq_lens, imgs) in enumerate(train_loader):
#         ans[i//(90*4)].append(units[0, :seq_lens[0]].tolist())
    
#     num_successful_decode = 0
#     for i, (units, padding_masks, seq_lens, imgs) in enumerate(val_loader):
#         if i%4 == 0:
#             imgs = imgs.to(device)
#             decoded_units = model.decode(word_map['<start>'], word_map['<end>'], image=imgs)
#             if decoded_units in ans[i//(20*4)]:
#                 num_successful_decode += 1
    
#     return num_successful_decode / (20*20)


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
    
    max_len = config["i2u"]["max_len"]
    model = TransformerVAEwithCNN(len(word_map), config["u2u"]["d_embed"], max_len = max_len).to(device)
    reconstruct_loss = nn.CrossEntropyLoss(reduction="sum").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["u2u"]["lr"])
    scheduler = get_lr_schedule(optimizer, num_warmup_epochs=config["u2u"]["warmup_epoch1"], d_model=config["u2u"]["d_model"])

    max_val_accuracy = 0
    # writer = SummaryWriter(f"../../model/U2U/transformer_vit1/log")
    # sys.stdout = Logger("../../model/U2U/transformer_vit1/vit1_log.txt")
    for epoch in range(config["u2u"]["epoch1"]):
        loss, accuracy = train(model, reconstruct_loss, train_loader, optimizer, device, kl_weight=config["u2u"]["kl_weight"], epoch = epoch)
        #sys.stdout = Logger("../../model/U2U/transformer_vit1/vit1_log.txt")
        print("epoch", epoch, "loss", loss, "accuracy", accuracy, flush=True)
        # val_accuracy = validate(model, device, word_map)
        val_loss, val_accuracy = validate(model, reconstruct_loss, val_loader, optimizer, device, kl_weight=config["u2u"]["kl_weight"], epoch = epoch)
        #sys.stdout = Logger("../../model/U2U/transformer_vit1/vit1_log.txt")
        print("epoch", epoch, "val accuracy", val_accuracy, flush=True)
        if max_val_accuracy < val_accuracy:
            max_val_accuracy = val_accuracy
            torch.save(model.state_dict(), config["u2u"]["path1"])
        scheduler.step()


if __name__ == "__main__":
    with open('../../config.yml') as yml:
        config = yaml.safe_load(yml)
    
    main(config)