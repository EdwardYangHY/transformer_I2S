import datetime
from glob import glob
import json
import os
import sys
import argparse
import gym
import numpy as np
import socket
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
import yaml

from datasets import CaptionDataset_transformer
from utils_synthesize import load_tacotron2, load_tacotron2_hubert, load_hifigan, load_asr
from utils_synthesize import load_i2u, load_i2u_all, seq2words, u2s, u2s_hubert, s2t

global device, is_debug
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_debug = True if sys.gettrace() else False
# is_debug = False
#------------------------------------------------------------------------------------------------

tts_model_path = "../../saved_model/U2S/outdir_komatsu_hubert_22050/checkpoint_9000_warmstart"
tacotron_model = load_tacotron2(model_path=tts_model_path, max_decoder_step = 500, sr = 22050, vocab_size = 102)

# tts_model_path = "/net/papilio/storage2/yhaoyuan/transformer_I2S/gslm_models/u2S/HuBERT_KM100_tts_checkpoint_best.pt"
# max_decoder_steps = 500
# code_dict_path = "/net/papilio/storage2/yhaoyuan/transformer_I2S/gslm_models/u2S/HuBERT_KM100_code_dict"
# tacotron_model, tts_dataset = load_tacotron2_hubert(model_path=tts_model_path, code_dict_path=code_dict_path, max_decoder_steps=max_decoder_steps)

### HifiGAN model for 22050 hz speech.
hifigan_checkpoint_path = "../../hifigan/LJ_FT_T2_V3/generator_v3"
hifigan_model = load_hifigan(hifigan_checkpoint_path, device)

# asr_checkpoint_path = config["asr"]["model_path"]
asr_checkpoint_path = "../../saved_model/ASR/wav2vec2-synthesize"
# asr_checkpoint_path = "../../saved_model/ASR/wav2vec2-base-tuned/checkpoint-3000"

asr_model, asr_processor = load_asr(asr_checkpoint_path, device)

#------------------------------------------------------------------------------------------------

def test(model, config):
    i_want = 0
    success = 0
    gt = []
    ans = []
    i = 0

    try:
        dir_name = config["i2u"]["dir_name"]
    except:
        dir_name = config["I2U"]["data_folder"]
        config["i2u"] = {}
        config["i2u"]["captions_per_image"]=4
        config["i2u"]["min_word_freq"]=1
    # dir_name = "komatsu_4_captions_256_hubert"

    # Data parameters
    # data_folder = f'../../data/processed/{dir_name}/'  # folder with data files saved by create_input_files.py
    if os.path.exists(f'../../data/processed/{dir_name}/'):
        data_folder = f'../../data/processed/{dir_name}/'  # folder with data files saved by create_input_files.py
    elif os.path.exists(f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}/'):
        data_folder = f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}/'
    else:
        raise ValueError(f"Dir: {dir_name} doesn't exist. Please check.")


    data_name = f'coco_{str(config["i2u"]["captions_per_image"])}_cap_per_img_{str(config["i2u"]["min_word_freq"])}_min_word_freq'  # base name shared by data files
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])    
    transform = transforms.Compose([normalize])
    val_loader = torch.utils.data.DataLoader(
            CaptionDataset_transformer(data_folder, data_name, 'VAL', transform=transform),
            batch_size=1, shuffle=False, num_workers=10, pin_memory=True)

    with torch.no_grad():
        for imgs, caps, caplens, padding_mask, all_caps, all_padding_mask in tqdm(val_loader):
            i+=1
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            caplens = caplens.squeeze()
            padding_mask = padding_mask.to(device)
            all_caps = all_caps.to(device)
            all_padding_mask = all_padding_mask.to(device)

            # get sentence_embedding:
            caplens = caplens.unsqueeze(0)
            mu = model.get_mu(caps, caplens, padding_mask)

            imgs, gx = model.image_encoder(imgs)
            flatten_imgs = imgs.reshape(-1, imgs.size(1)*imgs.size(2))
            # if flatten_imgs.dim() == 2:
            #     flatten_imgs = flatten_imgs.unsqueeze(0)
            # if mu.dim() == 2:
            #     mu = mu.unsqueeze(0)
            action = torch.cat([flatten_imgs, mu], dim=1)

            seqs = model.decode(action=action, start_unit=word_map["<start>"], end_unit=word_map["<end>"], max_len=150, beam_size=10)
            words = seq2words(seq=seqs, rev_word_map=rev_word_map, special_words=special_words)
            ### This is for Tacotron2 trained by ourselves
            audio = u2s(
                words=words,
                tacotron2_model=tacotron_model,
                hifigan_model=hifigan_model,
                device=device
                )
            # audio = u2s_hubert(
            #     words=words,
            #     tacotron2_model=tacotron_model,
            #     tts_dataset=tts_dataset,
            #     hifigan_model=hifigan_model,
            #     device=device
            #     )
            trans = s2t(audio=audio, asr_model=asr_model, asr_processor=asr_processor, device=device)

            seqs_gt = [int(x) for x in caps[0,:]]
            words_gt = seq2words(seq=seqs_gt, rev_word_map=rev_word_map, special_words=special_words)
            audio_gt = u2s(
                words=words_gt,
                tacotron2_model=tacotron_model,
                hifigan_model=hifigan_model,
                device=device
                )
            # audio_gt = u2s_hubert(
            #     words=words_gt,
            #     tacotron2_model=tacotron_model,
            #     tts_dataset=tts_dataset,
            #     hifigan_model=hifigan_model,
            #     device=device
            #     )
            trans_gt = s2t(audio=audio_gt, asr_model=asr_model, asr_processor=asr_processor, device=device)

            # if "I WANT" in trans_gt.upper():
            #     i_want += 1
            #     if "I WANT" in trans.upper():
            #         success += 1
            
            if "want" in trans_gt.lower():
                i_want += 1
                if "want" in trans.lower():
                    success += 1
            
            ans.append(trans)
            gt.append(trans_gt)
            
            # if is_debug:
            #     if i > 10:
            #         break
            # else:
            #     pass
            if i > 200:
                break

        rate = success/i_want
    return rate, gt, ans

def main(model_path):
    # is_debug = False
    if is_debug:
        pass

    global word_map, rev_word_map, special_words, i2u_model
    if os.path.isdir(model_path):
        i2u_model, word_map, rev_word_map, special_words, model_config = load_i2u_all(model_path, \
            haoyuan_model=True, show_config=True)
    elif os.path.isfile(model_path):
        i2u_model, word_map, rev_word_map, special_words, model_config = load_i2u_all(model_path, \
            word_map_path="../../data/processed/WORDMAP_HUBERT.json", haoyuan_model=False, \
            show_config=True)
    else:
        raise ValueError("model_path should be a file or a directory")

    i2u_model.to(device)
    
    I_want_acc, gt_list, ans_list = test(i2u_model, model_config)

    # with open(model_path + f"/val_recognition_results_sentence_no_incremental.txt", "w") as f:
    #         f.write("%-20s\t\t%-20s\n"%("I want Accuracy", f"{I_want_acc}"))
    #         f.write("%-20s\t\t%-20s\n"%("Ground Truth", "Answer")+ "-"*100+"\n")
    #         for i in range(len(ans_list)):
    #             # f.write(f"{names[i]} \t {ans_list[i]} \t {count_list[i]} \n ")
    #             f.write("%-20s\t\t%-20s\n"%(f"{gt_list[i]}", f"{ans_list[i]}"))
    try:
        with open(model_path + f"/val_recognition_results_sentence_no_incremental.txt", "w") as f:
                f.write("%-20s\t\t%-20s\n"%("Embedding Accuracy", f"{I_want_acc}"))
                f.write("%-20s\t\t%-20s\n"%("Ground Truth", "Answer")+ "-"*100+"\n")
                for i in range(len(ans_list)):
                    # f.write(f"{names[i]} \t {ans_list[i]} \t {count_list[i]} \n ")
                    f.write("%-20s\t\t%-20s\n"%(f"{gt_list[i]}", f"{ans_list[i]}"))
    except:
        with open(os.path.abspath(model_path+os.path.sep+"..") + f"/val_recognition_results_sentence_no_incremental.txt", "w") as f:
                f.write("%-20s\t\t%-20s\n"%("Embedding Accuracy", f"{I_want_acc}"))
                f.write("%-20s\t\t%-20s\n"%("Ground Truth", "Answer")+ "-"*100+"\n")
                for i in range(len(ans_list)):
                    # f.write(f"{names[i]} \t {ans_list[i]} \t {count_list[i]} \n ")
                    f.write("%-20s\t\t%-20s\n"%(f"{gt_list[i]}", f"{ans_list[i]}"))

if __name__ == "__main__":
    model_paths = [
        # "../../saved_model/I2U/komatsu_4_captions_224_hubert/Prefix_ViT_all_0.11_BLEU",
        # "../../saved_model/I2U/komatsu_4_captions_224_hubert/Prefix_CNN",
        # "../../saved_model/I2U/komatsu_4_captions_224_hubert/Prefix_ViT_6_layer_uLM",
        # "../../saved_model/I2U/komatsu_4_captions_224_hubert_5_percent/ICASSP_arch_uLM_6_layers_100_epochs",
        # "../../saved_model/I2U/komatsu_4_captions_224_hubert/Prefix_ViT_6_layer_uLM_6*8_SE_first_lr_10-3",
        # "../../saved_model/I2U/komatsu_4_captions_224_hubert/Prefix_ViT_6_layer_uLM_6*8_SE_first", 
        # "../../saved_model/I2U/komatsu_4_captions_224_hubert/Prefix_ViT_6_layer_uLM_6*8_SE_pool",
        # "../../saved_model/I2U/komatsu_4_captions_224_hubert/Prefix_ResNet_6_layer_uLM",
        "../../saved_model/I2U/komatsu_4_captions_224_hubert/23-07-08_15:27:33_ICASSP_pikaia19/i2u_with_sentence_embedding.pt"
        ]
    for model_path in model_paths:
        main(model_path)