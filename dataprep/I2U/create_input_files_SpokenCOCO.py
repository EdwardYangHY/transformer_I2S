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
    # Check dir_name exists or not, empty or not
    dir_name = "SpokenCOCO_5_captions_hubert_256"

    if os.path.exists(f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}'):
        assert not os.listdir(f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}'), "Exist dir, not empty. Choose a new name."
    else:
        os.mkdir(f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}')

    max_len = 200
    # max_len = 100

    base_path = "/net/tateha/storage2/database/misc/SpokenCOCO/SpokenCOCO/"
    train_split_json = "SpokenCOCO_train_augmented.json"
    val_split_json = "SpokenCOCO_val_augmented.json"

    with open(base_path+train_split_json, "r") as f:
        train_split = json.load(f)
    with open(base_path+val_split_json, "r") as f:
        val_split = json.load(f)

    def get_captions(data_split):
        img_paths = []
        hubert_captions = []
        for item in data_split["data"]:
            hubert_caps = []
            for cap in item["captions"]:
                # print(cap)
                hubert_caps.append(cap["hubert_cap"])
            img_paths.append(base_path + item["image"])
            hubert_captions.append(hubert_caps)
        return img_paths, hubert_captions

    train_image_paths, train_image_captions = get_captions(train_split)
    val_image_paths, val_image_captions = get_captions(val_split)
    # def get_captions(img_paths, paired_data):
    #     img_captions = []
    #     for i in tqdm(range(len(img_paths))):
    #         captions = [wav_captions[wav_base + one_wav] for one_wav in paired_data[i]["audio"]]
    #         img_captions.append(captions)
    #     assert len(img_captions) == len(img_paths), "Unmatched images/captions"
    #     return img_captions

    # train_image_captions = get_captions(train_image_paths, img_wav_data["data"]["train"])
    # val_image_captions = get_captions(val_image_paths, img_wav_data["data"]["val"])
    # test_image_captions = get_captions(test_image_paths, img_wav_data["data"]["test"])
    
    

    word_freq = Counter()
    for captions in train_image_captions:
        for caption in captions:
            word_freq.update(caption)

    # 
    print(f"saving /net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}")
    with open(f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}/train_image_paths.pickle', 'wb') as f:
        pickle.dump(train_image_paths, f)
    with open(f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}/train_image_captions.pickle', 'wb') as f:
        pickle.dump(train_image_captions, f)
    with open(f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}/val_image_paths.pickle', 'wb') as f:
        pickle.dump(val_image_paths, f)
    with open(f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}/val_image_captions.pickle', 'wb') as f:
        pickle.dump(val_image_captions, f)
    with open(f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}/test_image_paths.pickle', 'wb') as f:
        pickle.dump(val_image_paths, f)
    with open(f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}/test_image_captions.pickle', 'wb') as f:
        pickle.dump(val_image_captions, f)
    with open(f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}/word_freq.pickle', 'wb') as f:
        pickle.dump(word_freq, f)



    # import sys
    sys.path.append('../../egs/I2U/')
    from utils_i2u import create_input_files


    create_input_files(dataset='coco',
                    karpathy_json_path='dataset/caption/dataset_coco.json',
                    image_folder='dataset/caption_data/',
                    captions_per_image=5,
                    min_word_freq=1,
                    output_folder=f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}/',
                    # dir_name= dir_name,
                    max_len = max_len)

if __name__ == "__main__":
    main()