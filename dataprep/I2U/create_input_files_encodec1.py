import glob
from collections import Counter
import pickle
from gtts import gTTS
from pydub import AudioSegment
import librosa
import os
import pydub
import numpy as np
import math
from tqdm import tqdm
import torch
import json
import sys
# sys.path.append('../../egs/S2U/')
# from run_utils import load_audio_model_and_state
# from steps.unit_analysis import (get_feats_codes,DenseAlignment)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from dataloaders.utils import compute_spectrogram

sys.path.append('../egs/I2U/')
from utils import create_input_files

import yaml
from random import choice

with open('../../config.yml', 'r') as yml:
    config = yaml.safe_load(yml)

train_json = config["data"]["train_json"]
valid_json = config["data"]["valid_json"]
test_json = config["data"]["test_json"]
max_len = config["data"]["max_len"]

dir_name = config["i2u"]["dir_name"]

print("Load train data from: {} ".format(config["data"]["train_json"]))
print("Captions per image: {} ".format(config["i2u"]["captions_per_image"]))
print("Saving data to: {} ".format("../../data/I2U/processed/" + dir_name))

if not os.path.exists("../../data/I2U/processed/" + dir_name):
    os.makedirs("../../data/I2U/processed/" + dir_name)
if not os.path.exists("../../model/I2U/" + dir_name):
    os.makedirs("../../model/I2U/" + dir_name)

encodec_base_path = "/net/papilio/storage2/yhaoyuan/encodec/"
with open(encodec_base_path + 'train_image_paths.pickle', 'rb') as f:
    train_image_paths = pickle.load(f)
with open(encodec_base_path + 'train_image_captions.pickle', 'rb') as f:
    train_image_captions = pickle.load(f)
with open(encodec_base_path + 'val_image_paths.pickle', 'rb') as f:
    val_image_paths = pickle.load(f)
with open(encodec_base_path + 'val_image_captions.pickle', 'rb') as f:
    val_image_captions = pickle.load(f)
with open(encodec_base_path + 'test_image_paths.pickle', 'rb') as f:
    test_image_paths = pickle.load(f)
with open(encodec_base_path + 'test_image_captions.pickle', 'rb') as f:
    test_image_captions = pickle.load(f)
with open(encodec_base_path + 'word_freq.pickle', 'rb') as f:
    word_freq = pickle.load(f)

for captions in train_image_captions:
    del captions[1]
for captions in val_image_captions:
    del captions[1]
for captions in test_image_captions:
    del captions[1]

with open(f'../../data/I2U/processed/{dir_name}/train_image_paths.pickle', 'wb') as f:
    pickle.dump(train_image_paths, f)
with open(f'../../data/I2U/processed/{dir_name}/train_image_captions.pickle', 'wb') as f:
    pickle.dump(train_image_captions, f)
with open(f'../../data/I2U/processed/{dir_name}/val_image_paths.pickle', 'wb') as f:
    pickle.dump(val_image_paths, f)
with open(f'../../data/I2U/processed/{dir_name}/val_image_captions.pickle', 'wb') as f:
    pickle.dump(val_image_captions, f)
with open(f'../../data/I2U/processed/{dir_name}/test_image_paths.pickle', 'wb') as f:
    pickle.dump(test_image_paths, f)
with open(f'../../data/I2U/processed/{dir_name}/test_image_captions.pickle', 'wb') as f:
    pickle.dump(test_image_captions, f)

with open(f'../../data/I2U/processed/{dir_name}/word_freq.pickle', 'wb') as f:
    pickle.dump(word_freq, f)

create_input_files(dataset='coco',
                   karpathy_json_path='dataset/caption/dataset_coco.json', #ignored
                   image_folder='dataset/caption_data/',                   #ignored
                   captions_per_image=config["i2u"]["captions_per_image"],
                   min_word_freq=config["i2u"]["min_word_freq"],
                   output_folder=f'../../data/I2U/processed/{dir_name}/',
                   max_len=max_len) # change max_len, used to be 100

