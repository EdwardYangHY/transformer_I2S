# An script written to test the I2S model by
# Combining I2U, U2S and ASR

import datetime
from glob import glob
import json
import os
import sys
import yaml
from tqdm import tqdm
import argparse

import numpy as np
import h5py
from PIL import Image
import resampy
import torch
from torchvision import transforms

# from utils_i2u import *
from utils_synthesize import *
from judge_asr import judge_ans

sys.path.append("./models")
# from models import models_modified
from models import TransformerConditionedLM
from models_modified import TransformerSentenceLM_FixedImg_gated # TransformerSentenceLM_FixedImg
from models_modified import TransformerSentenceLM_FixedImg_Pool
from models_prompt import TransformerPrefixLM, prefix_Transformer
from incremental_decoder import incremental_prefix_Transformer, TransformerSentenceLM_FixedImg_Pool_incremental
# config path需要更改
is_debug = True if sys.gettrace() else False


def load_i2u(checkpoint_path, **model_params):
    params = checkpoint_path.split("/")[-2].split("_")
    if "gated" in params:
        model = TransformerSentenceLM_FixedImg_gated(**model_params)
    else:
        # model = TransformerSentenceLM_FixedImg(**model_params)
        model = TransformerSentenceLM_FixedImg_Pool_incremental(**model_params)
    model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    return model

def load_i2u_prev(checkpoint_path, **model_params):
    model = TransformerConditionedLM(**model_params)
    model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    return model

def load_i2u_prefix(checkpoint_path, **model_params):
    model = prefix_Transformer(**model_params)
    model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    return model

def load_i2u_increment(checkpoint_path, **model_params):
    model = incremental_prefix_Transformer(**model_params)
    model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    return model

def load_images(data_path, split):
    image_hdf5 = glob(data_path+f"/{split}*.hdf5")[0]
    image_names = glob(data_path+f"/{split}*.json")[0]
    h = h5py.File(image_hdf5, 'r')
    images = h['images']
    with open(image_names, "r") as f:
        names = json.load(f)
    return images, names

def get_transformed_img(img, transform): # -> torch.tensor
    # img = img.transpose(2, 0, 1) # (224, 224, 3) -> (3, 224, 224)
    img = torch.FloatTensor(img / 255.)
    if transform is not None:
        img = transform(img)
    return img.to(device)

def evaluate(model_path, word_map_path):

    with open('../../config.yml') as yml:
        config = yaml.safe_load(yml)
    
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### use pretrained u2s from gslm ###
    tts_model_path = "/net/papilio/storage2/yhaoyuan/transformer_I2S/gslm_models/u2S/HuBERT_KM100_tts_checkpoint_best.pt"
    max_decoder_steps = 500
    code_dict_path = "/net/papilio/storage2/yhaoyuan/transformer_I2S/gslm_models/u2S/HuBERT_KM100_code_dict"
    hifigan_checkpoint_path = "../../hifigan/LJ_FT_T2_V3/generator_v3"
    tacotron_model, tts_datasets = load_tacotron2_hubert(model_path=tts_model_path, code_dict_path=code_dict_path, max_decoder_steps=max_decoder_steps)
    asr_checkpoint_path = config["asr"]["model_path"]

    hifigan_model = load_hifigan(hifigan_checkpoint_path, device)
    asr_model, asr_processor = load_asr(asr_checkpoint_path, device)

    # --------------------------------------------------------------------------------

    # Load I2U:

    config_path = glob(model_path + "/config*.yml")[0]
    model_checkpoint = glob(model_path+"/*BEST*.tar")[0]

    # Load word map (word2ix)
    global word_map, rev_word_map, special_words
    with open(word_map_path) as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
    special_words = {"<unk>", "<start>", "<end>", "<pad>"}
    
    with open(config_path, 'r') as yml:
        model_config = yaml.safe_load(yml)

    model_params = model_config["i2u"]["model_params"]
    model_params['vocab_size'] = len(word_map)
    model_params['refine_encoder_params'] = model_config["i2u"]["refine_encoder_params"]


    i2u_model = load_i2u(model_checkpoint, **model_params)
    # i2u_model = load_i2u_prev(model_checkpoint, **model_params)
    # i2u_model = load_i2u_prefix(model_checkpoint, **model_params)
    # i2u_model = load_i2u_increment(model_checkpoint, **model_params)
    i2u_model.eval()
    i2u_model.to(device)

    image_resolution = 256 #224

    image_data_path = f"../../data/RL/{str(image_resolution)}"
    # for split in ["TRAIN", "VAL", "TEST"]:
    # for split in ["TRAIN"]:
    for split in ["VAL", "TEST"]:
        imgs, names = load_images(image_data_path, split)
        if is_debug:
            names = names[:10]

        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transform = None

        ans_list = []
        count_list = []
        count = 0
        count_name = 0

        for i in tqdm(range(len(names)), desc=f"Getting {split} Results"):
            img = get_transformed_img(imgs[i], transform)
            # img = get_transformed_img(imgs[i], transform=None)
            img = img.unsqueeze(0)
            name = names[i]
            # Test Action to Speech ---------------------------------------------------------------
            # action, gx = i2u_model.image_encoder(img)
            # action = action.flatten().unsqueeze(0)
            # seqs = i2u_model.decode(action=action, start_unit=word_map["<start>"], end_unit=word_map["<end>"], max_len=150, beam_size=10)
            # -------------------------------------------------------------------------------------
            seqs = i2u_model.decode(image=img, start_unit=word_map["<start>"], end_unit=word_map["<end>"], max_len=150, beam_size=10)
            # -------------------------------------------------------------------------------------
            try:
                words = seq2words(seq=seqs, rev_word_map=rev_word_map, special_words=special_words)
                # audio = u2s(
                #     words=words,
                #     tacotron2_model=tacotron_model,
                #     hifigan_model=hifigan_model,
                #     device=device
                #     )
                audio = u2s_hubert(
                    words=words,
                    tacotron2_model=tacotron_model,
                    tts_dataset=tts_datasets,
                    hifigan_model=hifigan_model,
                    device=device
                    )
                
                trans = s2t(audio=audio, asr_model=asr_model, asr_processor=asr_processor, device=device)
            except:
                trans = "U2S not successful."
            # trans = u2s2t(seq=seqs, tacotron2_model=tacotron_model, generator=hifigan_model, processor=asr_processor, asr_model=asr_model)
            
            right_ans, right_name =  judge_ans(trans, name)
            ans_list.append(trans)
            count_list.append(right_ans)
            if right_ans:
                count += 1
            if right_name:
                count_name += 1

        with open(model_path + f"/{split}_recognition_results_{image_resolution}_incremental_debug.txt", "w") as f:
            f.write("%-20s\t\t%-20s\n"%("Recognition Accuracy", f"{count/len(names)}"))
            f.write("%-20s\t\t%-20s\n"%("Recog Name Accuracy", f"{count_name/len(names)}")+ "-"*100 +"\n")
            f.write("%-20s\t\t%-50s\t\t%-20s\n"%("Image Name", "Synthesized Answer", "Right Answer?")+ "-"*100+"\n")
            for i in range(len(names)):
                # f.write(f"{names[i]} \t {ans_list[i]} \t {count_list[i]} \n ")
                f.write("%-20s\t\t%-50s\t\t%-20s\n"%(f"{names[i]}", f"{ans_list[i]}", f"{count_list[i]}"))

def main():
    # model_paths = ["../../saved_model/I2U/origin_5_captions_256_hubert/prefix_resolution_8_tune_image"]
    model_paths = ["../../saved_model/I2U/origin_4_captions_256_hubert_sentence/23-06-11_17:57:29_uLM_sentence"] 
    # model_paths = ["../../saved_model/I2U/origin_5_captions_256_hubert/hubert_baseline"]
    word_map_paths = ["../../data/processed/origin_5_captions_256_hubert/WORDMAP_coco_5_cap_per_img_1_min_word_freq.json"]

    for model_path, word_map_path in zip(model_paths, word_map_paths):
        evaluate(model_path, word_map_path)

if __name__ == "__main__":
    main()