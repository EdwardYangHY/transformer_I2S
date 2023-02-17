import json

import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from models import TransformerVAEwithCNN
from dataset import UnitImageDataset

with open('../../config.yml') as yml:
    config = yaml.safe_load(yml)

word_map_path=config["i2u"]["wordmap"]
# Load word map (word2ix)
with open(word_map_path) as j:
    word_map = json.load(j)

# I2U
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerVAEwithCNN(len(word_map), config["u2u"]["d_embed"])
model.load_state_dict(torch.load(config["u2u"]["path2"]))
model.eval()
model.to(device)

train_loader = DataLoader(
    UnitImageDataset(config["u2u"]["data_folder"], config["u2u"]["data_name"], 'TRAIN'),
    batch_size=1,
    shuffle=False,
    num_workers=config["u2u"]["num_workers"],
    pin_memory=True,
    )
# 20*90*4

embed = torch.zeros((20, 4, 90, config["u2u"]["d_embed"]))

for i, (units, padding_masks, seq_lens, imgs) in enumerate(tqdm(train_loader)):
    units = units.to(device)
    padding_masks = padding_masks.to(device)
    seq_lens = seq_lens.to(device)
    imgs = imgs.to(device)
    
    z = model.encode(units, padding_masks, seq_lens)
    
    food_type = i//(90*4)
    desc_type = i%4
    image_num = (i%(90*4))//4

    embed[food_type, desc_type, image_num, :] = z

mean = embed.mean(dim=2)

print(mean)