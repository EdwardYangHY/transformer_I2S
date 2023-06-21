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
from datasets import CaptionDataset_transformer
import yaml
from utils_synthesize import load_tacotron2, load_hifigan,load_asr, load_i2u, seq2words, u2s, s2t

global device, is_debug
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# is_debug = True if sys.gettrace() else False
is_debug = False

#------------------------------------------------------------------------------------------------
# model_path = "../../saved_model/I2U/komatsu_4_captions_224_hubert/Prefix_ViT"
# word_map_path = None
# if word_map_path is None:
#     word_map_path = "../../data/processed/origin_5_captions_256_hubert/WORDMAP_coco_5_cap_per_img_1_min_word_freq.json"
# else:
#     word_map_path = word_map_path

# config_path = glob(model_path + "/config*.yml")[0]
# model_checkpoint = glob(model_path+"/*BEST*.tar")[0]
# if not os.path.isfile(config_path):
#     raise ValueError(f"{config_path} invalid. Please check the model path.")
# if not os.path.isfile(model_checkpoint):
#     raise ValueError(f"{model_checkpoint} invalid. Please check the model path.")

# # Load word map (word2ix)
# global word_map, rev_word_map, special_words, i2u_model
# with open(word_map_path) as j:
#     word_map = json.load(j)
# rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
# special_words = {"<unk>", "<start>", "<end>", "<pad>"}
# with open(config_path, 'r') as yml:
#     model_config = yaml.safe_load(yml)

# dir_name = model_config["i2u"]["dir_name"]

# model_params = model_config["i2u"]["model_params"]
# model_params['vocab_size'] = len(word_map)
# model_params['refine_encoder_params'] = model_config["i2u"]["refine_encoder_params"]

# d_embed = 0
# if model_params["use_sentence_encoder"]:
#     d_embed = model_params["sentence_embed"]

# # i2u_model = load_i2u_codec(model_checkpoint, **model_params)
# i2u_model = load_i2u(model_checkpoint, **model_params)
# i2u_model.eval()
# i2u_model.to(device)
# prefix_i2u_model = i2u_model
#------------------------------------------------------------------------------------------------

tts_model_path = "../../saved_model/U2S/outdir_komatsu_hubert_22050/checkpoint_9000_warmstart"
tacotron_model = load_tacotron2(model_path=tts_model_path, max_decoder_step = 500, sr = 22050, vocab_size = 102)

### HifiGAN model for 22050 hz speech.
hifigan_checkpoint_path = "../../hifigan/LJ_FT_T2_V3/generator_v3"
hifigan_model = load_hifigan(hifigan_checkpoint_path, device)

# asr_checkpoint_path = config["asr"]["model_path"]
asr_checkpoint_path = "../../saved_model/ASR/wav2vec2-synthesize"
asr_model, asr_processor = load_asr(asr_checkpoint_path, device)

#------------------------------------------------------------------------------------------------

def test(model, config):
    i_want = 0
    success = 0
    gt = []
    ans = []
    i = 0

    dir_name = config["i2u"]["dir_name"]
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

            # -------------------------------------------------------------------------------------------
            # mu_prefix = prefix_i2u_model.get_mu(caps, caplens, padding_mask)
            # -------------------------------------------------------------------------------------------

            # reparamatize
            # log_std = torch.full_like(mu, 0.1).log()
            # eps = torch.randn_like(log_std)  # (batch, sentence_embed)
            # mu = mu + eps*log_std.exp()  # (batch, sentence_embed)

            imgs, gx = model.image_encoder(imgs)
            flatten_imgs = imgs.reshape(-1, imgs.size(1)*imgs.size(2))
            action = torch.cat([flatten_imgs, mu], dim=1)

            # -------------------------------------------------------------------------------------------
            # action_prefix = torch.cat([flatten_imgs, mu_prefix], dim=1)
            # seqs_prefix = model.decode(action=action_prefix, start_unit=word_map["<start>"], end_unit=word_map["<end>"], max_len=150, beam_size=10)
            # words_prefix = seq2words(seq=seqs_prefix, rev_word_map=rev_word_map, special_words=special_words)
            # ### This is for Tacotron2 trained by ourselves
            # audio_prefix = u2s(
            #     words=words_prefix,
            #     tacotron2_model=tacotron_model,
            #     hifigan_model=hifigan_model,
            #     device=device
            #     )
            # trans_prefix = s2t(audio=audio_prefix, asr_model=asr_model, asr_processor=asr_processor, device=device)
            # -------------------------------------------------------------------------------------------

            seqs = model.decode(action=action, start_unit=word_map["<start>"], end_unit=word_map["<end>"], max_len=150, beam_size=10)
            words = seq2words(seq=seqs, rev_word_map=rev_word_map, special_words=special_words)
            ### This is for Tacotron2 trained by ourselves
            audio = u2s(
                words=words,
                tacotron2_model=tacotron_model,
                hifigan_model=hifigan_model,
                device=device
                )
            trans = s2t(audio=audio, asr_model=asr_model, asr_processor=asr_processor, device=device)

            seqs_gt = [int(x) for x in caps[0,:]]
            words_gt = seq2words(seq=seqs_gt, rev_word_map=rev_word_map, special_words=special_words)
            audio_gt = u2s(
                words=words_gt,
                tacotron2_model=tacotron_model,
                hifigan_model=hifigan_model,
                device=device
                )
            trans_gt = s2t(audio=audio_gt, asr_model=asr_model, asr_processor=asr_processor, device=device)

            if "I WANT" in trans_gt.upper():
                i_want += 1
                if "I WANT" in trans.upper():
                    success += 1
            
            ans.append(trans)
            gt.append(trans_gt)
            
            # if is_debug:
            #     if i > 10:
            #         break
            # else:
            #     pass
            if i > 100:
                break

        rate = success/i_want
    return rate, gt, ans

def main(model_path):
    # is_debug = False
    if is_debug:
        # if socket.gethostname() == "pikaia25":
        #     model_path = "../../saved_model/I2U/komatsu_4_captions_256_hubert/Prefix_baseline_BLEU_12.5"
        # elif socket.gethostname() == "pikaia28":
        #     model_path = "../../saved_model/I2U/komatsu_4_captions_256_hubert/Codec_baseline_BLEU_12"
        # else:
        #     model_path = "../../saved_model/I2U/komatsu_4_captions_256_hubert/Codec_baseline_BLEU_12_7*7_no_tune"
        # model_path = "../../saved_model/I2U/komatsu_4_captions_256_hubert/Codec_baseline_BLEU_12_7*7_no_tune"
        pass
        # model_path = "../../saved_model/I2U/komatsu_4_captions_256_hubert/Prefix_baseline_BLEU_12.5"

    word_map_path = None
    if word_map_path is None:
        word_map_path = "../../data/processed/origin_5_captions_256_hubert/WORDMAP_coco_5_cap_per_img_1_min_word_freq.json"
    else:
        word_map_path = word_map_path

    config_path = glob(model_path + "/config*.yml")[0]
    model_checkpoint = glob(model_path+"/*BEST*.tar")[0]
    if not os.path.isfile(config_path):
        raise ValueError(f"{config_path} invalid. Please check the model path.")
    if not os.path.isfile(model_checkpoint):
        raise ValueError(f"{model_checkpoint} invalid. Please check the model path.")

    # Load word map (word2ix)
    global word_map, rev_word_map, special_words, i2u_model
    with open(word_map_path) as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
    special_words = {"<unk>", "<start>", "<end>", "<pad>"}

    with open(config_path, 'r') as yml:
        model_config = yaml.safe_load(yml)

    dir_name = model_config["i2u"]["dir_name"]

    model_params = model_config["i2u"]["model_params"]
    model_params['vocab_size'] = len(word_map)
    model_params['refine_encoder_params'] = model_config["i2u"]["refine_encoder_params"]

    d_embed = 0
    if model_params["use_sentence_encoder"]:
        d_embed = model_params["sentence_embed"]

    # i2u_model = load_i2u_codec(model_checkpoint, **model_params)
    i2u_model = load_i2u(model_checkpoint, **model_params)
    i2u_model.eval()
    i2u_model.to(device)
    
    I_want_acc, gt_list, ans_list = test(i2u_model, model_config)

    with open(model_path + f"/val_recognition_results_sentence_no_incremental.txt", "w") as f:
            f.write("%-20s\t\t%-20s\n"%("I want Accuracy", f"{I_want_acc}"))
            #f.write("%-20s\t\t%-20s\n"%("Recog Name Accuracy", f"{count_name/len(names)}")+ "-"*100 +"\n")
            f.write("%-20s\t\t%-20s\n"%("Ground Truth", "Answer")+ "-"*100+"\n")
            for i in range(len(ans_list)):
                # f.write(f"{names[i]} \t {ans_list[i]} \t {count_list[i]} \n ")
                f.write("%-20s\t\t%-20s\n"%(f"{gt_list[i]}", f"{ans_list[i]}"))

if __name__ == "__main__":
    model_paths = [
        "../../saved_model/I2U/komatsu_4_captions_224_hubert/Codec_baseline_ViT",
        # "../../saved_model/I2U/komatsu_4_captions_224_hubert/Prefix_ViT",
        # "../../saved_model/I2U/komatsu_4_captions_224_ResDAVEnet/Prefix_CNN_ResDAVE",
        # "../../saved_model/I2U/komatsu_4_captions_224_hubert/Codec_baseline_224",
        # "../../saved_model/I2U/komatsu_4_captions_256_hubert/Prefix_baseline_BLEU_12.5",
        # "../../saved_model/I2U/komatsu_4_captions_256_hubert/Codec_baseline_new",
        # "../../saved_model/I2U/komatsu_4_captions_256_hubert/Codec_baseline_BLEU_12",
        # "../../saved_model/I2U/komatsu_4_captions_256_hubert/Codec_baseline_BLEU_12_7*7_no_tune"
        ]
    for model_path in model_paths:
        main(model_path)

            