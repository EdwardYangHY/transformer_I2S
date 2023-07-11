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
# from utils_synthesize import *
from utils_synthesize import load_tacotron2, load_tacotron2_hubert, load_hifigan, load_asr
from utils_synthesize import load_i2u, load_i2u_all, seq2words, u2s, u2s_hubert, s2t
from judge_asr import judge_ans

# sys.path.append("./models")
# # from models import models_modified
# from models import TransformerConditionedLM
# from models_modified import TransformerSentenceLM_FixedImg_gated # TransformerSentenceLM_FixedImg
# from models_modified import TransformerSentenceLM_FixedImg_Pool
# from models_prompt import TransformerPrefixLM, prefix_Transformer
# config path需要更改
is_debug = True if sys.gettrace() else False

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def load_i2u(checkpoint_path, **model_params):
#     params = checkpoint_path.split("/")[-2].split("_")
#     if "gated" in params:
#         model = TransformerSentenceLM_FixedImg_gated(**model_params)
#     else:
#         # model = TransformerSentenceLM_FixedImg(**model_params)
#         model = TransformerSentenceLM_FixedImg_Pool(**model_params)
#     model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
#     return model

# def load_i2u_prev(checkpoint_path, **model_params):
#     model = TransformerConditionedLM(**model_params)
#     model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
#     return model

# def load_i2u_prefix(checkpoint_path, **model_params):
#     model = prefix_Transformer(**model_params)
#     model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
#     return model

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

# ### Synthesize ###
# tts_model_path = "../../saved_model/U2S/outdir_komatsu_hubert_22050/checkpoint_9000_warmstart"
# tacotron_model = load_tacotron2(model_path=tts_model_path, max_decoder_step = 500, sr = 22050, vocab_size = 102)

### Record with VC model ###
tts_model_path = "../../saved_model/U2S/outdir_VC_hubert_22050_102_warm/checkpoint_33000"
tacotron_model = load_tacotron2(model_path=tts_model_path, max_decoder_step = 600, sr = 22050, vocab_size = 102)

# ### LJSpeech by GSLM ###
# tts_model_path = "/net/papilio/storage2/yhaoyuan/transformer_I2S/gslm_models/u2S/HuBERT_KM100_tts_checkpoint_best.pt"
# max_decoder_steps = 500
# code_dict_path = "/net/papilio/storage2/yhaoyuan/transformer_I2S/gslm_models/u2S/HuBERT_KM100_code_dict"
# tacotron_model, tts_dataset = load_tacotron2_hubert(model_path=tts_model_path, code_dict_path=code_dict_path, max_decoder_steps=max_decoder_steps)

### HifiGAN model for 22050 hz speech.
hifigan_checkpoint_path = "../../hifigan/LJ_FT_T2_V3/generator_v3"
hifigan_model = load_hifigan(hifigan_checkpoint_path, device)

# asr_checkpoint_path = config["asr"]["model_path"]
### ASR model tuned on Record Speech###
asr_checkpoint_path = "/net/papilio/storage2/yhaoyuan/LAbyLM/model/ASR/wav2vec2-base-tuned/checkpoint-3000"
asr_model, asr_processor = load_asr(asr_checkpoint_path, device)

def evaluate(model_path):

    # with open('../../config.yml') as yml:
    #     config = yaml.safe_load(yml)
    
    global word_map, rev_word_map, special_words, i2u_model
    i2u_model, word_map, rev_word_map, special_words, model_config = load_i2u_all(model_path, \
        haoyuan_model=True, show_config=True)
    i2u_model.to(device)

    image_resolution = 256 #224

    image_data_path = f"../../data/RL/{str(image_resolution)}"
    # for split in ["TRAIN", "VAL", "TEST"]:
    # for split in ["TRAIN"]:
    for split in ["VAL", "TEST"]:
        imgs, names = load_images(image_data_path, split)
        if is_debug:
            # names = names[:10]
            pass

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
                audio = u2s(
                    words=words,
                    tacotron2_model=tacotron_model,
                    hifigan_model=hifigan_model,
                    device=device
                    )
                # audio = u2s_hubert(
                #     words=words,
                #     tacotron2_model=tacotron_model,
                #     tts_dataset=tts_datasets,
                #     hifigan_model=hifigan_model,
                #     device=device
                #     )
                
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

        with open(model_path + f"/{split}_recognition_results_{image_resolution}_VC_U2S_22050.txt", "w") as f:
            f.write("%-20s\t\t%-20s\n"%("Recognition Accuracy", f"{count/len(names)}"))
            f.write("%-20s\t\t%-20s\n"%("Recog Name Accuracy", f"{count_name/len(names)}")+ "-"*100 +"\n")
            f.write("%-20s\t\t%-50s\t\t%-20s\n"%("Image Name", "Synthesized Answer", "Right Answer?")+ "-"*100+"\n")
            for i in range(len(names)):
                # f.write(f"{names[i]} \t {ans_list[i]} \t {count_list[i]} \n ")
                f.write("%-20s\t\t%-50s\t\t%-20s\n"%(f"{names[i]}", f"{ans_list[i]}", f"{count_list[i]}"))


def main():
    # model_paths = ["../../saved_model/I2U/origin_5_captions_256/baseline_lr-3_no_LM",
    #                "../../saved_model/I2U/origin_5_captions_256/lr-3_uLM"]
    # word_map_paths = ["../../data/processed/origin_5_captions_256/WORDMAP_coco_5_cap_per_img_1_min_word_freq.json",
    #                   "../../data/processed/origin_5_captions_256/WORDMAP_coco_5_cap_per_img_1_min_word_freq.json"]
    model_paths = ["../../saved_model/I2U/origin_5_captions_256_hubert/prefix_resolution_8_tune_image"]
    
    for model_path in model_paths:
        evaluate(model_path)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '--model_path', type=str,
    #                     help='directory to the saved i2u model')
    # args = parser.parse_args()
    # main(args.model_path)
    main()