from glob import glob
import json
import os
import sys

import gym
import numpy as np
from PIL import Image
from imageio import imread
import resampy
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import yaml


sys.path.append("../..")
import hifigan
from hifigan.env import AttrDict
from hifigan.models import Generator

sys.path.append("../U2S")
from hparams import create_hparams
from train import load_model
from text import text_to_sequence


sys.path.append("../I2U/models/")
# from models import TransformerSentenceLM
from models_modified import TransformerSentenceLM_FixedImg, TransformerSentenceLM_FixedImg_gated
from models_k import TransformerVAEwithCNN


class Unit2Text(object):
    def __init__(self, i2u_model, u2s_model, vocoder_generator, asr_processor, asr_model, word_map, device):
        self.i2u_model = i2u_model
        self.u2s_model = u2s_model
        self.generator = vocoder_generator
        self.asr_processor = asr_processor
        self.asr_model = asr_model
        self.device = device
        self.word_map = word_map
        self.rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
        self.special_words = {"<unk>", "<start>", "<end>", "<pad>"}

    def i2u2s2t(self, action):
        action = torch.from_numpy(action).unsqueeze(0).to(self.device)
        seq = self.i2u_model.decode(self.word_map['<start>'], self.word_map['<end>'], action=action, max_len=130, beam_size=5)
        words = [self.rev_word_map[ind] for ind in seq if self.rev_word_map[ind] not in self.special_words]
        sequence = np.array(text_to_sequence(' '.join(words), ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        try:
            _, mel_outputs_postnet, _, _ = self.u2s_model.inference(sequence)
            with torch.no_grad():
                y_g_hat = self.generator(mel_outputs_postnet)
                audio = y_g_hat.squeeze()
            audio = audio.cpu().numpy().astype(np.float64)
            # audio = resampy.resample(audio, 22050, 16000)
            # s2t
            input_values = self.asr_processor(audio, sampling_rate=16000, return_tensors="pt").input_values.float()
            logits = self.asr_model(input_values.to(self.device)).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.asr_processor.decode(predicted_ids[0])
        except RuntimeError as e:
            transcription = ""
            print(e, flush=True)
        return transcription


def load_i2u_checkpoint(checkpoint_path, **model_params):
    params = checkpoint_path.split("/")[-2].split("_")
    if "gated" in params:
        model = TransformerSentenceLM_FixedImg_gated(**model_params)
    else:
        model = TransformerSentenceLM_FixedImg(**model_params)
    model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    return model



def load_I2U(model_path, word_map, device):
    config_path = glob(model_path+"/config*.yml")[0]
    model_checkpoint = glob(model_path+"/*BEST*.tar")[0 ]

    with open(config_path, 'r') as yml:
        model_config = yaml.safe_load(yml)

    model_params = model_config["i2u"]["model_params"]
    model_params['vocab_size'] = len(word_map)
    model_params['refine_encoder_params'] = model_config["i2u"]["refine_encoder_params"]

    sentence_encoder = load_i2u_checkpoint(model_checkpoint, **model_params)
    sentence_encoder.eval()
    sentence_encoder.to(device)
    return sentence_encoder

def load_I2U_komatsu(checkpoint_path, vocab_size, sentence_embedding_size, device):
    sentence_encoder = TransformerVAEwithCNN(vocab_size, sentence_embedding_size, max_len = 100)
    sentence_encoder.load_state_dict(torch.load(checkpoint_path))
    sentence_encoder.eval()
    sentence_encoder.to(device)
    return sentence_encoder

def load_U2S(checkpoint_path, device = None):
    hparams = create_hparams()
    hparams.sampling_rate = 22050

    # tacotron2
    # checkpoint_path = "../../saved_model/U2S_synthesize/checkpoint_40000"
    tacotron2_model = load_model(hparams)
    tacotron2_model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    tacotron2_model.cuda().eval()
    return tacotron2_model

def load_vocoder(checkpoint_path, device):
    # checkpoint_file = config["u2s"]['hifigan']
    config_file = os.path.join(os.path.split(checkpoint_path)[0], 'config.json')
    with open(config_file) as f:
            data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)
    generator = Generator(h).to(device)
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint_dict['generator'])
    generator.eval()
    generator.remove_weight_norm()
    return generator

def load_ASR(checkpoint_path, device):
    processor = Wav2Vec2Processor.from_pretrained(checkpoint_path)
    asr_model = Wav2Vec2ForCTC.from_pretrained(checkpoint_path).to(device)
    return processor, asr_model

def i2u2s2t(action, i2u_model, tacotron2_model, generator, processor, asr_model, word_map, device):
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
    special_words = {"<unk>", "<start>", "<end>", "<pad>"}

    action = torch.from_numpy(action).unsqueeze(0).to(device)
    seq = i2u_model.decode(word_map['<start>'], word_map['<end>'], action=action, max_len=130, beam_size=5)
    words = [rev_word_map[ind] for ind in seq if rev_word_map[ind] not in special_words]
    sequence = np.array(text_to_sequence(' '.join(words), ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    try:
        _, mel_outputs_postnet, _, _ = tacotron2_model.inference(sequence)
        with torch.no_grad():
            y_g_hat = generator(mel_outputs_postnet)
            audio = y_g_hat.squeeze()
        audio = audio.cpu().numpy().astype(np.float64)
        # audio = resampy.resample(audio, 22050, 16000)
        # s2t
        input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.float()
        logits = asr_model(input_values.to(device)).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
    except RuntimeError as e:
        transcription = ""
        print(e, flush=True)
    return transcription