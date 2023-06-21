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
import sys
# sys.path.append('../../egs/S2U/')
# from run_utils import load_audio_model_and_state
# from steps.unit_analysis import (get_feats_codes,DenseAlignment)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from dataloaders.utils import compute_spectrogram

import yaml
import json

def main():
    with open('../../config.yml', 'r') as yml:
        config = yaml.safe_load(yml)

    # Check dir_name exists or not, empty or not
    dir_name = config["i2u"]["dir_name"]

    # output_folder
    # output_folder = "../../data/processed/"
    output_folder = "/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/"
    if os.path.exists(output_folder+dir_name):
        assert not os.listdir(output_folder+dir_name), "Exist dir, not empty. Choose a new name."
    else:
        os.mkdir(output_folder+dir_name)

    # if os.path.exists(f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}'):
    #     assert not os.listdir(f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}'), "Exist dir, not empty. Choose a new name."
    # else:
    #     os.mkdir(f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}')
    
    max_len = config["data"]["max_len"]
    # max_len = 100

    img_wav_data = config["data"]["dataset_json"]
    # img_wav_data = '/net/papilio/storage2/yhaoyuan/transformer_I2S/data/food_dataset_origin_shuffle.json'
    with open(img_wav_data, "r") as f:
        img_wav_data = json.load(f)

    img_base = img_wav_data["image_base_path"]
    wav_base = img_wav_data["audio_base_path"]

    wav_captions = config["data"]["wav_captions"]
    # wav_captions = "/net/papilio/storage2/yhaoyuan/LAbyLM/data/I2U/audios_24k_captions.json"
    with open(wav_captions, "r") as f:
        wav_captions = json.load(f)


    train_image_paths = [img_base + one_data["image"] for one_data in img_wav_data["data"]["train"]]
    val_image_paths = [img_base + one_data["image"] for one_data in img_wav_data["data"]["val"]]
    test_image_paths = [img_base + one_data["image"] for one_data in img_wav_data["data"]["test"]]


    def get_captions(img_paths, paired_data):
        img_captions = []
        for i in tqdm(range(len(img_paths))):
            captions = [wav_captions[wav_base + one_wav] for one_wav in paired_data[i]["audio"]]
            img_captions.append(captions)
        assert len(img_captions) == len(img_paths), "Unmatched images/captions"
        return img_captions

    train_image_captions = get_captions(train_image_paths, img_wav_data["data"]["train"])
    val_image_captions = get_captions(val_image_paths, img_wav_data["data"]["val"])
    test_image_captions = get_captions(test_image_paths, img_wav_data["data"]["test"])

    word_freq = Counter()
    for captions in train_image_captions:
        for caption in captions:
            word_freq.update(caption)

    # 
    print(f"saving at {output_folder}{dir_name}")
    with open(f'{output_folder}{dir_name}/train_image_paths.pickle', 'wb') as f:
        pickle.dump(train_image_paths, f)
    with open(f'{output_folder}{dir_name}/train_image_captions.pickle', 'wb') as f:
        pickle.dump(train_image_captions, f)
    with open(f'{output_folder}{dir_name}/val_image_paths.pickle', 'wb') as f:
        pickle.dump(val_image_paths, f)
    with open(f'{output_folder}{dir_name}/val_image_captions.pickle', 'wb') as f:
        pickle.dump(val_image_captions, f)
    with open(f'{output_folder}{dir_name}/test_image_paths.pickle', 'wb') as f:
        pickle.dump(test_image_paths, f)
    with open(f'{output_folder}{dir_name}/test_image_captions.pickle', 'wb') as f:
        pickle.dump(test_image_captions, f)
    with open(f'{output_folder}{dir_name}/word_freq.pickle', 'wb') as f:
        pickle.dump(word_freq, f)



    # import sys
    sys.path.append('../../egs/I2U/')
    from utils_i2u import create_input_files


    create_input_files(dataset='coco',
                    karpathy_json_path='dataset/caption/dataset_coco.json',
                    image_folder='dataset/caption_data/',
                    captions_per_image=config["i2u"]["captions_per_image"],
                    min_word_freq=config["i2u"]["min_word_freq"],
                    output_folder=f'{output_folder}{dir_name}/',
                    # dir_name= dir_name,
                    max_len = max_len)

if __name__ == "__main__":
    main()