import glob
from collections import Counter
import os
import numpy as np
from tqdm import tqdm
import torch
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import yaml
import json

def read_dict(path):
    with open(path, "r") as f:
        unit_dict = json.load(f)
    return unit_dict

def cut_len(caption, limit_len) -> list:
    """
        if the caption surpasses limited_len, it will cut the caption into small pieces.
    """
    return [caption[i:i+limit_len] for i in range(0,len(caption),limit_len)]

def main():
    data_name = "Libri_Light_small_hubert_256"

    if os.path.exists(f'../../data/processed/{data_name}'):
        assert not os.listdir(f'../../data/processed/{data_name}'), "Exist dir, not empty. Choose a new name."
    else:
        os.mkdir(f'../../data/processed/{data_name}')

    # max_len = 150
    limit_len = 250
    min_word_freq = 1
    # spc_train_path = "../../data/I2U/SpokenCOCO_train_units_dict.json"
    # spc_val_path = "../../data/I2U/SpokenCOCO_val_units_dict.json"
    # lbs_train_path = "../../data/I2U/LibriSpeech_train_units_dict.json"
    # lbs_val_path = "../../data/I2U/LibriSpeech_val_units_dict.json"

    libri_light = "../../data/libri_light/libri_light_small_hbcaps.json"
    data = read_dict(libri_light)
    train_caps = []
    val_caps = []
    max_len = 0
    
    for i, (k, v) in enumerate(data.items()):
        max_len = len(v) if len(v) > max_len else max_len
        if len(v) > limit_len:
            temp_caps = cut_len(v, limit_len)
        else:
            temp_caps = [v]
        if i <= 0.95 * len(data):
            for c in temp_caps:
                train_caps.append(c)
        else:
            for c in temp_caps:
                val_caps.append(c)

    # word_freq = Counter()
    # for caption in train_caps:
    #     # for caption in captions:
    #     word_freq.update(caption)

    # words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    # word_map = {k: v + 1 for v, k in enumerate(words)}
    # word_map['<unk>'] = len(word_map) + 1
    # word_map['<start>'] = len(word_map) + 1
    # word_map['<end>'] = len(word_map) + 1
    # word_map['<pad>'] = 0
    
    output_folder=f'../../data/processed/{data_name}/'
    base_filename = "coco" + '_' + str(1) + '_cap_per_img_' + str(1) + '_min_word_freq'
    
    # with open(os.path.join(output_folder, 'WORDMAP_HUBERT.json'), 'w') as j:
    #     json.dump(word_map, j)

    with open("/net/papilio/storage2/yhaoyuan/transformer_I2S/data/processed/Libri_Light_small_hubert_512/WORDMAP_HUBERT.json", "r") as f:
        word_map = json.load(f)

    for caps, split in [(train_caps, "TRAIN"),
                       (val_caps, "VAL")]:
        enc_captions = []
        caplens = []
        print(f"Process {split}")
        for cap in tqdm(caps):
            enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in cap] + [
                        word_map['<end>']] + [word_map['<pad>']] * (limit_len - len(cap))

            # Find caption lengths
            c_len = len(cap) + 2

            enc_captions.append(enc_c)
            caplens.append(c_len)

        with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
            json.dump(enc_captions, j)

        with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
            json.dump(caplens, j)
        
if __name__ == "__main__":
    main()
