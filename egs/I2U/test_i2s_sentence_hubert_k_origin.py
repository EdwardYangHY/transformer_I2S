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
import resampy
import torch
# import torch.backends.cudnn as cudnn
# import torch.optim
# import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from datasets import CaptionDataset_transformer
import yaml
from utils_synthesize import load_tacotron2, load_hifigan,load_asr, load_i2u, seq2words, u2s, s2t
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

sys.path.append("/net/papilio/storage2/yhaoyuan/transformer_I2S/egs/I2U/models")
from models_k import TransformerVAEwithCNN_Modified
from utils_synthesize import load_i2u

# sys.path.append("../..")
# import hifigan
# from hifigan.env import AttrDict
# from hifigan.models import Generator

sys.path.append("../U2S")
# from hparams import create_hparams
# from train import load_model
from text import text_to_sequence

# sys.path.append("../I2U/models")
# # from models import TransformerSentenceLM
# from models_k import TransformerVAEwithCNN

with open('../../config.yml') as yml:
    config = yaml.safe_load(yml)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_debug = True if sys.gettrace() else False

# I2U
print("Preparing I2U model")
# word_map_path="../../data/processed/synthesize_speech/WORDMAP_coco_4_cap_per_img_1_min_word_freq.json"
word_map_path="../../data/processed/WORDMAP_ResDAVEnet.json"
# Load word map (word2ix)
with open(word_map_path) as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
special_words = {"<unk>", "<start>", "<end>", "<pad>"}


# I2U
# config["u2u"] = {}
# config["u2u"]["d_embed"] = 8
# sentence_encoder = TransformerVAEwithCNN_Modified(len(word_map), config["u2u"]["d_embed"], max_len=100)
# sentence_encoder.load_state_dict(torch.load("../../saved_model/U2U/transformer_cnn2_synthesize.pt"))
# # sentence_encoder.load_state_dict() ../../saved_model/I2U/komatsu_4_captions_256_hubert/TransVAEwCNN_Komatsu/bleu-4_BEST_checkpoint_coco_4_cap_per_img_1_min_word_freq_gpu.pth.tar
# sentence_encoder.eval()
# sentence_encoder.to(device)


# model_path = "../../saved_model/I2U/synthesize_speech/Codec_baseline"
model_path = "../../saved_model/I2U/komatsu_4_captions_224_ResDAVEnet/Prefix_ViT_ResDAVE"
config_path = glob(model_path + "/config*.yml")[0]
model_checkpoint = glob(model_path+"/*BEST*.tar")[0]
if not os.path.isfile(config_path):
    raise ValueError(f"{config_path} invalid. Please check the model path.")
if not os.path.isfile(model_checkpoint):
    raise ValueError(f"{model_checkpoint} invalid. Please check the model path.")

# Load word map (word2ix)
# global word_map, rev_word_map, special_words, i2u_model
# with open(word_map_path) as j:
#     word_map = json.load(j)
# rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
# special_words = {"<unk>", "<start>", "<end>", "<pad>"}

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
sentence_encoder = i2u_model

##--------------------------------------------------------------------------------------------------
tts_model_path = "../../saved_model/U2S_synthesize/checkpoint_40000"
tacotron_model = load_tacotron2(model_path=tts_model_path, max_decoder_step = 1000, sr = 22050, vocab_size = 1024)
tacotron2_model = tacotron_model

### HifiGAN model for 22050 hz speech.
hifigan_checkpoint_path = "../../hifigan/LJ_FT_T2_V3/generator_v3"
hifigan_model = load_hifigan(hifigan_checkpoint_path, device)
generator = hifigan_model

# asr_checkpoint_path = config["asr"]["model_path"]
asr_checkpoint_path = "../../saved_model/ASR/wav2vec2-synthesize"
asr_model, asr_processor = load_asr(asr_checkpoint_path, device)
processor = asr_processor
##--------------------------------------------------------------------------------------------------

def u2s2t(seqs_gt):
    words_gt = seq2words(seq=seqs_gt, rev_word_map=rev_word_map, special_words=special_words)
    audio_gt = u2s(
        words=words_gt,
        tacotron2_model=tacotron_model,
        hifigan_model=hifigan_model,
        device=device
        )
    # audio_gt = resampy.resample(audio_gt, 22050, 16000)
    trans_gt = s2t(audio=audio_gt, asr_model=asr_model, asr_processor=asr_processor, device=device)
    return trans_gt

def i2u2s2t(action):
    # action = torch.from_numpy(action).unsqueeze(0).to(device)
    seq = sentence_encoder.decode(start_unit=word_map['<start>'],  end_unit=word_map['<end>'], action=action, beam_size=10)
    transcription = u2s2t(seq)
    return transcription

# def i2u2s2t(action):
#     # action = torch.from_numpy(action).unsqueeze(0).to(device)
#     seq = sentence_encoder.decode(word_map['<start>'], word_map['<end>'], action=action, beam_size=1)
#     words = [rev_word_map[ind] for ind in seq if rev_word_map[ind] not in special_words]
#     sequence = np.array(text_to_sequence(' '.join(words), ['english_cleaners']))[None, :]
#     sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
#     try:
#         _, mel_outputs_postnet, _, _ = tacotron2_model.inference(sequence)
#         with torch.no_grad():
#             y_g_hat = generator(mel_outputs_postnet)
#             audio = y_g_hat.squeeze()
#         audio = audio.cpu().numpy().astype(np.float64)
#         audio = resampy.resample(audio, 22050, 16000)
#         # s2t
#         input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.float()
#         logits = asr_model(input_values.to(device)).logits
#         predicted_ids = torch.argmax(logits, dim=-1)
#         transcription = processor.decode(predicted_ids[0])
#     except RuntimeError as e:
#         transcription = ""
#         print(e, flush=True)
#     return transcription

def test(model, config):
    i_want = 0
    success = 0
    gt = []
    ans = []
    i = 0

    # dir_name = "synthesize_speech"
    dir_name = "komatsu_4_captions_224_ResDAVEnet"
    # dir_name = config["i2u"]["dir_name"]
    # dir_name = "komatsu_4_captions_256_hubert"

    # Data parameters
    # data_folder = f'../../data/processed/{dir_name}/'  # folder with data files saved by create_input_files.py
    if os.path.exists(f'../../data/processed/{dir_name}/'):
        data_folder = f'../../data/processed/{dir_name}/'  # folder with data files saved by create_input_files.py
    elif os.path.exists(f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}/'):
        data_folder = f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}/'
    else:
        raise ValueError(f"Dir: {dir_name} doesn't exist. Please check.")


    data_name = 'coco_4_cap_per_img_1_min_word_freq'  # base name shared by data files
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

            # reparamatize
            # log_std = torch.full_like(mu, 0.1).log()
            # eps = torch.randn_like(log_std)  # (batch, sentence_embed)
            # mu = mu + eps*log_std.exp()  # (batch, sentence_embed)

            # imgs, gx = model.image_encoder(imgs)
            imgs, gx = model.image_encoder(imgs)
            # imgs = model.cnn(imgs)
            flatten_imgs = imgs.reshape(-1, imgs.size(1)*imgs.size(2))
            action = torch.cat([flatten_imgs, mu], dim=1)

            trans = i2u2s2t(action)
            
            seqs_gt = [int(x) for x in caps[0,:]]
            trans_gt = u2s2t(seqs_gt)

            if "I WANT" in trans_gt.upper():
                i_want += 1
                if "I WANT" in trans.upper():
                    success += 1
            
            ans.append(trans)
            gt.append(trans_gt)
            
            if i > 100:
                break

        rate = success/i_want
    return rate, gt, ans

def main(model_path):
    
    I_want_acc, gt_list, ans_list = test(sentence_encoder, None)

    with open(f"val_recognition_results_sentence_Prefix_ResDAVEnet.txt", "w") as f:
            f.write("%-20s\t\t%-20s\n"%("I want Accuracy", f"{I_want_acc}"))
            #f.write("%-20s\t\t%-20s\n"%("Recog Name Accuracy", f"{count_name/len(names)}")+ "-"*100 +"\n")
            f.write("%-20s\t\t%-20s\n"%("Ground Truth", "Answer")+ "-"*100+"\n")
            for i in range(len(ans_list)):
                # f.write(f"{names[i]} \t {ans_list[i]} \t {count_list[i]} \n ")
                f.write("%-20s\t\t%-20s\n"%(f"{gt_list[i]}", f"{ans_list[i]}"))

if __name__ == "__main__":
    model_paths = [
        "../../saved_model/I2U/komatsu_4_captions_256_hubert/TransVAEwCNN_Komatsu",
        # "../../saved_model/I2U/komatsu_4_captions_224_hubert/Codec_baseline_224",
        # "../../saved_model/I2U/komatsu_4_captions_256_hubert/Prefix_baseline_BLEU_12.5",
        # "../../saved_model/I2U/komatsu_4_captions_256_hubert/Codec_baseline_new",
        # "../../saved_model/I2U/komatsu_4_captions_256_hubert/Codec_baseline_BLEU_12",
        # "../../saved_model/I2U/komatsu_4_captions_256_hubert/Codec_baseline_BLEU_12_7*7_no_tune"
        ]
    for model_path in model_paths:
        main(model_path)

            