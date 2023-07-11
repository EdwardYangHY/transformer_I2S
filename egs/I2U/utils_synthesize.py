
import sys
import os
import torch
import json
import numpy as np
import yaml
from glob import glob

sys.path.append("../..")
import hifigan
from hifigan.env import AttrDict
from hifigan.models import Generator

sys.path.append("../U2S")
# import U2S
from hparams import create_hparams
from train import load_model
from text import text_to_sequence

sys.path.append('../')
from gslm.unit2speech.tts_data import (
    TacotronInputDataset,
)
from gslm.unit2speech.utils import (
    load_quantized_audio_from_file,
    load_tacotron,
    load_waveglow,
    synthesize_audio,
)
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# sys.path.append("./models")
sys.path.append("/net/papilio/storage2/yhaoyuan/transformer_I2S/egs/I2U/models")
# from models import models_modified
from models import TransformerConditionedLM
# from models_modified import TransformerSentenceLM_FixedImg
from models_modified import TransformerSentenceLM_FixedImg_Pool, TransformerSentenceLM_FixedImg_gated
from models_prompt import TransformerPrefixLM, prefix_Transformer
from incremental_decoder import incremental_prefix_Transformer, TransformerSentenceLM_FixedImg_Pool_incremental
from models_ICASSP import ImageToUnit
from models_k import ImageToUnit as ImageToUnit_haoyuan

def load_tacotron2(model_path, max_decoder_step = None, sr = None, vocab_size = None):
    hparams = create_hparams()
    if vocab_size is not None:
        hparams.n_symbols = vocab_size
    if sr is not None:
        hparams.sampling_rate = sr
    if max_decoder_step is not None:
        hparams.max_decoder_steps = max_decoder_step
    checkpoint_path = model_path
    tacotron2_model = load_model(hparams)
    tacotron2_model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    tacotron2_model.cuda().eval()
    return tacotron2_model

def load_tacotron2_hubert(model_path, code_dict_path, max_decoder_steps):
    tacotron_model, sample_rate, hparams = load_tacotron(
        tacotron_model_path=model_path,
        max_decoder_steps=max_decoder_steps,
    )

    if not os.path.exists(hparams.code_dict):
        hparams.code_dict = code_dict_path
    tts_dataset = TacotronInputDataset(hparams)
    return tacotron_model, tts_dataset

def load_hifigan(checkpoint_path, device):
    checkpoint_file = checkpoint_path
    config_file = os.path.join(os.path.split(checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    # global h
    json_config = json.loads(data)
    h = AttrDict(json_config)
    generator = Generator(h).to(device)
    assert os.path.isfile(checkpoint_file)
    checkpoint_dict = torch.load(checkpoint_file, map_location=device)
    generator.load_state_dict(checkpoint_dict['generator'])
    generator.eval()
    generator.remove_weight_norm()
    return generator

def load_asr(model_path, device):
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    asr_model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
    asr_model.eval()
    return asr_model, processor

def load_i2u_codec(checkpoint_path, **model_params):
    params = checkpoint_path.split("/")[-2].split("_")
    if "gated" in params:
        model = TransformerSentenceLM_FixedImg_gated(**model_params)
        print("A codec gated decoder is loaded.")
    else:
        # model = TransformerSentenceLM_FixedImg(**model_params)
        # model = TransformerSentenceLM_FixedImg_Pool(**model_params)
        # print("A codec plain decoder is loaded.")
        model = TransformerSentenceLM_FixedImg_Pool_incremental(**model_params)
        print("A codec incremental decoder is loaded.")
    model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    return model

def load_i2u_prefix(checkpoint_path, **model_params):
    model = incremental_prefix_Transformer(**model_params)
    # model = prefix_Transformer(**model_params)
    model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    print("A prefix incremental decoder is loaded.")
    return model

def load_i2u(checkpoint_path, **model_params):
    model_name = checkpoint_path.split("/")[-2].split("_")[0]
    if model_name.lower() == "codec":
        return load_i2u_codec(checkpoint_path, **model_params)
    elif model_name.lower() == "prefix":
        return load_i2u_prefix(checkpoint_path, **model_params)
    else:
        raise ValueError("The model structure is not specified/implied by the model's path. \
            please us \"load_i2u_codec/prefix\" instead. ")
    return

# def load_i2u_all_(model_path):
#     dir_name = model_path.split("/")[-2]
#     if os.path.exists(f'/net/papilio/storage2/yhaoyuan/transformer_I2S/data/processed/{dir_name}/'):
#         data_folder = f'/net/papilio/storage2/yhaoyuan/transformer_I2S/data/processed/{dir_name}/'  # folder with data files saved by create_input_files.py
#     elif os.path.exists(f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}/'):
#         data_folder = f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}/'
#     else:
#         raise ValueError(f"Dir: {dir_name} doesn't exist. Please check.")

#     word_map_path = glob(data_folder+"*WORDMAP*.json")[0]

#     config_path = glob(model_path + "/config*.yml")[0]
#     model_checkpoint = glob(model_path+"/*BEST*.tar")[0]
#     if not os.path.isfile(config_path):
#         raise ValueError(f"{config_path} invalid. Please check the model path.")
#     if not os.path.isfile(model_checkpoint):
#         raise ValueError(f"{model_checkpoint} invalid. Please check the model path.")

#     # Load word map (word2ix)
#     # global word_map, rev_word_map, special_words, i2u_model
#     with open(word_map_path) as j:
#         word_map = json.load(j)
#     rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
#     special_words = {"<unk>", "<start>", "<end>", "<pad>"}

#     config_path = glob(model_path + "/config*.yml")[0]
#     model_checkpoint = glob(model_path+"/*BEST*.tar")[0]
#     if not os.path.isfile(config_path):
#         raise ValueError(f"{config_path} invalid. Please check the model path.")
#     if not os.path.isfile(model_checkpoint):
#         raise ValueError(f"{model_checkpoint} invalid. Please check the model path.")

#     # Load word map (word2ix)
#     # global word_map, rev_word_map, special_words #, i2u_model
#     with open(word_map_path) as j:
#         word_map = json.load(j)
#     rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
#     special_words = {"<unk>", "<start>", "<end>", "<pad>"}

#     with open(config_path, 'r') as yml:
#         model_config = yaml.safe_load(yml)

#     dir_name = model_config["i2u"]["dir_name"]

#     model_params = model_config["i2u"]["model_params"]
#     model_params['vocab_size'] = len(word_map)
#     model_params['refine_encoder_params'] = model_config["i2u"]["refine_encoder_params"]

#     d_embed = 0
#     if model_params["use_sentence_encoder"]:
#         d_embed = model_params["sentence_embed"]

#     # i2u_model = load_i2u_codec(model_checkpoint, **model_params)
#     i2u_model = load_i2u(model_checkpoint, **model_params)
#     i2u_model.eval()
#     # i2u_model.to(device)
#     return i2u_model, word_map, rev_word_map, special_words

# def load_i2u(checkpoint_path, model_config_path, vocab_size):
#     with open(model_config_path, "r") as yml:
#         model_config = yaml.safe_load(yml)
#     model_params = model_config["i2u"]["model_params"]
#     model_params['vocab_size'] = vocab_size
#     model_params['refine_encoder_params'] = model_config["i2u"]["refine_encoder_params"]
    
#     model = TransformerSentenceLM_FixedImg_Pool(**model_params)
#     model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
#     return model

def load_i2u_all(model_path, word_map_path=None, haoyuan_model=True, show_config=False):
    if haoyuan_model:
        dir_name = model_path.split("/")[-2]
        if os.path.exists(f'/net/papilio/storage2/yhaoyuan/transformer_I2S/data/processed/{dir_name}/'):
            data_folder = f'/net/papilio/storage2/yhaoyuan/transformer_I2S/data/processed/{dir_name}/'  # folder with data files saved by create_input_files.py
        elif os.path.exists(f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}/'):
            data_folder = f'/net/papilio/storage6/yhaoyuan/SpeechCap/data/processed/{dir_name}/'
        else:
            raise ValueError(f"Dir: {dir_name} doesn't exist. Please check.")
        word_map_path = glob(data_folder+"*WORDMAP*.json")[0]

        config_path = glob(model_path + "/config*.yml")[0]
        model_checkpoint = glob(model_path+"/*BEST*.tar")[0]
        if not os.path.isfile(config_path):
            raise ValueError(f"{config_path} invalid. Please check the model path.")
        if not os.path.isfile(model_checkpoint):
            raise ValueError(f"{model_checkpoint} invalid. Please check the model path.")

        # Load word map (word2ix)
        # global word_map, rev_word_map, special_words, i2u_model
        with open(word_map_path) as j:
            word_map = json.load(j)
        rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
        special_words = {"<unk>", "<start>", "<end>", "<pad>"}

        config_path = glob(model_path + "/config*.yml")[0]
        model_checkpoint = glob(model_path+"/*BEST*.tar")[0]
        if not os.path.isfile(config_path):
            raise ValueError(f"{config_path} invalid. Please check the model path.")
        if not os.path.isfile(model_checkpoint):
            raise ValueError(f"{model_checkpoint} invalid. Please check the model path.")

        # Load word map (word2ix)
        # global word_map, rev_word_map, special_words #, i2u_model
        with open(word_map_path) as j:
            word_map = json.load(j)
        rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
        special_words = {"<unk>", "<start>", "<end>", "<pad>"}

        with open(config_path, 'r') as yml:
            model_config = yaml.safe_load(yml)

        # dir_name = model_config["i2u"]["dir_name"]

        model_params = model_config["i2u"]["model_params"]
        model_params['vocab_size'] = len(word_map)
        model_params['refine_encoder_params'] = model_config["i2u"]["refine_encoder_params"]

        d_embed = 0
        if model_params["use_sentence_encoder"]:
            d_embed = model_params["sentence_embed"]

        # i2u_model = load_i2u_codec(model_checkpoint, **model_params)
        i2u_model = load_i2u(model_checkpoint, **model_params)
        i2u_model.eval()
        # i2u_model.to(device)
    else:
        assert word_map_path is not None, "For Komatsu Model, we need a word_map file"
        with open(word_map_path) as j:
            word_map = json.load(j)
        rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
        special_words = {"<unk>", "<start>", "<end>", "<pad>"}
        # Komatsu model.
        dir_path = os.path.abspath(model_path + os.path.sep + "..")
        try:
            config_path = glob(dir_path + "/config*.yml")[0]
        except:
            config_path = None

        model_config = None
        try:
            if config_path is not None:
                with open(config_path, 'r') as yml:
                    model_config = yaml.safe_load(yml)
                i2u_model = ImageToUnit(word_map, max_len = model_config["I2U"]["max_len"])
            else:
                i2u_model = ImageToUnit(word_map)
            # config["I2U"]["max_len"]
            # i2u_model = ImageToUnit(word_map)
            i2u_model.load_state_dict(torch.load(model_path))
        except:
            if config_path is not None:
                with open(config_path, 'r') as yml:
                    model_config = yaml.safe_load(yml)
                i2u_model = ImageToUnit_haoyuan(word_map, max_len = model_config["I2U"]["max_len"])
            else:
                i2u_model = Image2Unit_haoyuan(word_map)
            # config["I2U"]["max_len"]
            # i2u_model = ImageToUnit(word_map)
            i2u_model.load_state_dict(torch.load(model_path))
        i2u_model.eval()
        print("A ImageToUnit model is loaded")

    if show_config and model_config is not None:
        return i2u_model, word_map, rev_word_map, special_words, model_config
    else:
        return i2u_model, word_map, rev_word_map, special_words


def load_u2s(device):
    with open('../../config.yml') as yml:
        config = yaml.safe_load(yml)

    tacotron_max_decoder_step = 1000 #config["u2s"]["max_decoder_steps"]
    tacotron_checkpoint_path = config["u2s"]["tacotron2"]
    hifigan_checkpoint_path = config["u2s"]['hifigan']
    asr_checkpoint_path = config["asr"]["model_path"]

    tacotron_model = load_tacotron2(tacotron_checkpoint_path, tacotron_max_decoder_step)
    hifigan_model = load_hifigan(hifigan_checkpoint_path, device)
    asr_model, asr_processor = load_asr(asr_checkpoint_path, device)

    return tacotron_model, hifigan_model, asr_model, asr_processor

def load_u2s_hubert(device):
    with open('../../config.yml') as yml:
        config = yaml.safe_load(yml)

    tts_model_path = "/net/papilio/storage2/yhaoyuan/transformer_I2S/gslm_models/u2S/HuBERT_KM100_tts_checkpoint_best.pt"
    max_decoder_steps = 500
    code_dict_path = "/net/papilio/storage2/yhaoyuan/transformer_I2S/gslm_models/u2S/HuBERT_KM100_code_dict"

    hifigan_checkpoint_path = config["u2s"]['hifigan']
    asr_checkpoint_path = config["asr"]["model_path"]

    # tacotron_model = load_tacotron(tacotron_checkpoint_path, tacotron_max_decoder_step)
    tacotron_model, tts_datasets = load_tacotron2_hubert(model_path=tts_model_path, code_dict_path=code_dict_path, max_decoder_steps=max_decoder_steps)
    hifigan_model = load_hifigan(hifigan_checkpoint_path, device)
    asr_model, asr_processor = load_asr(asr_checkpoint_path, device)

    return tacotron_model, tts_datasets, hifigan_model, asr_model, asr_processor

# --------------------------------------------------------------------------------------------------

def seq2words(seq, rev_word_map, special_words):
    return [rev_word_map[ind] for ind in seq if rev_word_map[ind] not in special_words]

def u2s(words, tacotron2_model, hifigan_model, device):
    # words = [rev_word_map[ind] for ind in seq if rev_word_map[ind] not in special_words]
    # print(words)
    sequence = np.array(text_to_sequence(' '.join(words), ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    _, mel_outputs_postnet, _, _ = tacotron2_model.inference(sequence)
    with torch.no_grad():
        x = mel_outputs_postnet.squeeze().to(device)
        y_g_hat = hifigan_model(mel_outputs_postnet)
        audio = y_g_hat.squeeze()
        # audio = audio * 32768.0
        # audio = audio.cpu().numpy().astype('int16')
        audio = audio.cpu().numpy().astype(np.float64)
    return audio

def synthesize_mel(model, inp, lab=None, strength=0.0):
    assert inp.size(0) == 1
    inp = inp.cuda()
    if lab is not None:
        lab = torch.LongTensor(1).cuda().fill_(lab)

    with torch.no_grad():
        _, mel, _, ali, has_eos = model.inference(inp, lab, ret_has_eos=True)
    return mel, has_eos

def u2s_hubert(words, tacotron2_model, tts_dataset, hifigan_model, device):
    quantized_units_str = " ".join(words)
    tts_input = tts_dataset.get_tensor(quantized_units_str)
    mel, has_eos = synthesize_mel(
        tacotron2_model,
        tts_input.unsqueeze(0),
    )
    with torch.no_grad():
        x = mel.squeeze().float()
        # x = torch.FloatTensor(x).to(device)
        y_g_hat = hifigan_model(x)
        audio = y_g_hat.squeeze()
        audio = audio * 32768.0
        # audio = audio.cpu().numpy().astype('int16')
        audio = audio.cpu().numpy().astype(np.float64)
    return audio

def s2t(audio, asr_processor, asr_model, device):
    input_values = asr_processor(audio, sampling_rate=16000, return_tensors="pt").input_values.float()
    logits = asr_model(input_values.to(device)).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = asr_processor.decode(predicted_ids[0])
    return transcription