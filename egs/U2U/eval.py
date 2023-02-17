import json
import statistics

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from models import TransformerVAEwithViT
from dataset import UnitImageDataset


def make_answer():
    import glob
    train_image_paths = glob.glob('../../data/I2U/image/*/train_number[123]/*.jpg')
    sorted_train_image_paths = sorted(glob.glob('../../data/I2U/image/*/train_number[123]/*.jpg'))
    for i, j in zip(train_image_paths, sorted_train_image_paths):
        assert i == j

    ans_mat = [[[], [], [], []] for _ in range(20)]

    train_loader = DataLoader(
        UnitImageDataset(config["u2u"]["data_folder"], config["u2u"]["data_name"], 'TRAIN'),
        batch_size=1,
        shuffle=False,
        num_workers=config["u2u"]["num_workers"],
        pin_memory=True,
        )
    
    for i, (units, padding_masks, seq_lens, imgs) in enumerate(train_loader):
        assert units.size(0) == 1
        ans_mat[i//(90*4)][i%4].append(units[0, :seq_lens[0]].cpu().tolist())
    print(ans_mat, flush=True)
    return ans_mat


def make_embedding():
    train_loader = DataLoader(
        UnitImageDataset(config["u2u"]["data_folder"], config["u2u"]["data_name"], 'TRAIN'),
        batch_size=1,
        shuffle=False,
        num_workers=config["u2u"]["num_workers"],
        pin_memory=True,
        )

    embed = torch.zeros((20, 4, 90, 8), device=device)

    for i, (units, padding_masks, seq_lens, imgs) in enumerate(train_loader):
        units = units.to(device)
        padding_masks = padding_masks.to(device)
        seq_lens = seq_lens.to(device)
        imgs = imgs.to(device)
        
        z = model.encode(units, padding_masks, seq_lens, deterministic=True, return_memory=False)
        
        food_type = i//(90*4)
        desc_type = i%4
        image_num = (i%(90*4))//4

        embed[food_type, desc_type, image_num, :] = z
    return embed


with open('../../config.yml') as yml:
    config = yaml.safe_load(yml)

word_map_path=config["i2u"]["wordmap"]
# Load word map (word2ix)
with open(word_map_path) as j:
    word_map = json.load(j)

# I2U
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerVAEwithViT(len(word_map), config["u2u"]["d_embed"])
model.load_state_dict(torch.load(config["u2u"]["path2"]))
model.eval()
model.to(device)

ans_mat = make_answer()
embed = make_embedding()

loader = DataLoader(
    UnitImageDataset(config["u2u"]["data_folder"], config["u2u"]["data_name"], 'TEST'),
    batch_size=1,
    shuffle=False,
    num_workers=config["u2u"]["num_workers"],
    pin_memory=True,
    )

res_dict = np.zeros((20, 10, 4))

res_dict_per_img = [0 for i in range(200)]
per_food = 10*4

for i, (units, padding_masks, seq_lens, imgs) in enumerate(loader):
    imgs = imgs.to(device)

    with torch.no_grad():
        img_feat = model.vit(imgs)

    for j in range(90):
        # z = torch.randn((1, 8), device=device)
        z = embed[i//per_food, i%4, j:j+1, :]
        assert z.size() == (1, 8)

        action = torch.cat([img_feat, z], dim=1)

        assert action.size() == (1, 768+8)

        decoded_units = model.decode(word_map['<start>'], word_map['<end>'], action=action)
        
        if decoded_units in ans_mat[i//per_food][0]:
            res_dict[i//per_food, (i%per_food)//4, 0] += 1
        elif decoded_units in ans_mat[i//per_food][1]:
            res_dict[i//per_food, (i%per_food)//4, 1] += 1
        elif decoded_units in ans_mat[i//per_food][2]:
            res_dict[i//per_food, (i%per_food)//4, 2] += 1
        elif decoded_units in ans_mat[i//per_food][3]:
            res_dict[i//per_food, (i%per_food)//4, 3] += 1
    
    if i%4 == 3:
        res_dict_per_img[i//4] = int(res_dict[i//per_food, (i%per_food)//4, 0]>0) + int(res_dict[i//per_food, (i%per_food)//4, 1]>0) + int(res_dict[i//per_food, (i%per_food)//4, 2]>0) + int(res_dict[i//per_food, (i%per_food)//4, 3]>0)
        print(i, res_dict_per_img, flush=True)


diversity = 0
for i in range(20):
    diversity_per_food = statistics.mean(res_dict_per_img[10*i:10*(i+1)])
    print(diversity_per_food)

    diversity += diversity_per_food
diversity = diversity / 20
print("diversity", diversity)