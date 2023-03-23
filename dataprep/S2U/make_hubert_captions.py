import argparse
import logging
import os

import numpy as np

import joblib
import soundfile as sf
import torch
import json
import yaml

import sys
sys.path.append("../../egs/")

from gslm.speech2unit.clustering.utils import (
    get_audio_files,
)
from gslm.speech2unit.pretrained.utils import (
    get_features,
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

feature_type = "hubert"
checkpoint_path = "/net/papilio/storage2/yhaoyuan/transformer_I2S/gslm_models/S2u/hubert_base_ls960.pt"
layer = 6
kmeans_model_path = "/net/papilio/storage2/yhaoyuan/transformer_I2S/gslm_models/S2u/HuBERT_100_km.bin"

kmeans_model = joblib.load(open(kmeans_model_path, "rb"))
kmeans_model.verbose = False

def RLE(seq):
    pred = []
    prev = -1
    for i in seq:
        if i != prev:
            pred.append(i)
            prev = i
        else:
            continue
    return pred

def RLE_str(seq):
    pred = []
    prev = -1
    for i in seq:
        if i != prev:
            pred.append(str(i))
            prev = i
        else:
            continue
    return pred

def make_units_and_save(manifest_path, save_name):

    file_paths = read_manifest(manifest_path)

    features_batch = get_features(
        feature_type=feature_type,
        checkpoint_path=checkpoint_path,
        layer=layer,
        manifest_path=manifest_path,
        sample_pct=1.0,
        flatten=False,
        channel_id=None,
    )
    
    predictions = []
    for i, feats in enumerate(features_batch):
        pred = kmeans_model.predict(feats)
        # pred_str = " ".join(str(p) for p in pred)
        predictions.append(pred)
    
    predictions_RLE = {}

    assert len(file_paths) == len(predictions)
    for i, prediction in enumerate(predictions):
        predictions_RLE[file_paths[i]] = RLE_str(predictions[i])
        # predictions_RLE.append(RLE_str(prediction))
    
    with open(save_name, "w") as f:
        json.dump(predictions_RLE, f)

def read_manifest(manifest_path):
    with open(manifest_path, "r") as f:
        data = f.readlines()
    file_paths = []
    for i in range(1, len(data)):
        file_name = data[0].strip("\n") + data[i].strip("\n")
        if os.path.isfile(file_name):
            file_paths.append(file_name)
        else:
            print(f"Not exist: {file_name}")
    return file_paths


def main():
    # data path
    # base_path = "/net/papilio/storage2/yhaoyuan/transformer_I2S/data/libri_light/"
    # manifest_path = "food_dataset_origin_shuffle_manifest.txt"
    # save_name = "food_dataset_origin_shuffle_hbcaps.json"
    
    # manifest_paths = []
    # save_names = []
    # manifest_paths.append(base_path + "food_dataset_origin_shuffle_manifest.txt")
    # save_names.append(base_path + "food_dataset_original_shuffle_hbcaps.json")

    # manifest_paths.append(base_path + "food_dataset_VC_shuffle_manifest.txt")
    # save_names.append(base_path + "food_dataset_VC_shuffle_hbcaps.json")
    
    # manifest_paths.append(base_path + "libri_light_small_manifest_1.txt")
    # save_names.append(base_path + "libri_light_small_hbcaps_1.json")
    # manifest_paths.append(base_path + "libri_light_small_manifest_2.txt")
    # save_names.append(base_path + "libri_light_small_hbcaps_2.json")
    # manifest_paths.append(base_path + "libri_light_small_manifest_3.txt")
    # save_names.append(base_path + "libri_light_small_hbcaps_3.json")
    # manifest_paths.append(base_path + "libri_light_small_manifest_4.txt")
    # save_names.append(base_path + "libri_light_small_hbcaps_4.json")

    manifest_paths = ["/net/papilio/storage2/yhaoyuan/transformer_I2S/data/food_dataset_gtts_shuffle_manifest.txt"]
    save_names = ["food_dataset_gtts_shuffle_hbcaps.json"]
    # manifest_path = "/net/papilio/storage2/yhaoyuan/transformer_I2S/data/food_dataset_VC_shuffle_manifest_short.txt"
    # manifest_path = "/net/papilio/storage2/yhaoyuan/transformer_I2S/data/libri_light_small_manifest.txt"

    # manifest_paths = []
    # manifest_paths.append("/net/papilio/storage2/yhaoyuan/transformer_I2S/data/food_dataset_origin_shuffle_manifest_short.txt")
    # manifest_paths.append("/net/papilio/storage2/yhaoyuan/transformer_I2S/data/food_dataset_VC_shuffle_manifest_short.txt")
    # manifest_paths.append("/net/papilio/storage2/yhaoyuan/transformer_I2S/data/libri_light_small_manifest.txt")
    # 

    for manifest_path, save_name in zip(manifest_paths, save_names):
        assert os.path.isfile(manifest_path)
        # print(manifest_path, save_name)
        make_units_and_save(manifest_path, save_name)

if __name__ == "__main__":
    main() 