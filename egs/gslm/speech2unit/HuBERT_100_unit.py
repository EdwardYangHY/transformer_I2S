import os

import numpy as np

import joblib
import soundfile as sf
import torch
import json
import sys
import librosa

sys.path.append('../..')
from gslm.speech2unit.pretrained.hubert_feature_reader import (
    HubertFeatureReader,
)

from gslm.unit2speech.tts_data import (
    TacotronInputDataset,
)
from gslm.unit2speech.utils import (
    load_quantized_audio_from_file,
    load_tacotron,
    load_waveglow,
    synthesize_audio,
)

# Load HuBERT
hubert_model = "/net/papilio/storage2/yhaoyuan/transformer_I2S/gslm_models/S2u/hubert_base_ls960.pt"
hubert_cluster_100 = "/net/papilio/storage2/yhaoyuan/transformer_I2S/gslm_models/S2u/HuBERT_100_km.bin"
feature_reader = HubertFeatureReader(checkpoint_path=hubert_model, layer=6)

# Load Cluster
kmeans_model = joblib.load(open(hubert_cluster_100, "rb"))
kmeans_model.verbose = False

def run_length_encoding(seq):
    pred = []
    prev = -1
    for i in seq:
        if i != prev:
            pred.append(i)
            prev = i
        else:
            continue
    return np.array(pred)

def get_units(wav_path):
    features = feature_reader.get_feats_(wav_path).cpu().numpy()
    pred = kmeans_model.predict(features)
    pred_RLE = run_length_encoding(pred)
    return pred_RLE