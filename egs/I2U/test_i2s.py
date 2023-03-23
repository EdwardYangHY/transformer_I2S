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
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

sys.path.append("../..")
import hifigan
from hifigan.env import AttrDict
from hifigan.models import Generator

sys.path.append("../U2S")
# import U2S
from hparams import create_hparams
from train import load_model
from text import text_to_sequence

sys.path.append("./models")
# from models import models_modified
from models import TransformerConditionedLM, PositionalEncoding
from models_modified import TransformerSentenceLM_FixedImg, TransformerSentenceLM_FixedImg_gated

# config path需要更改
is_debug = True if sys.gettrace() else False

name_dict={
    'apple': ['apple','apples'],
    'banana': ['banana','bananas'],
    'carrot': ['carrot','carrots'],
    'grape': ['grape','grapes'],
    'cucumber': ['cucumber','cucumbers'],
    'egg': ['egg','eggs'],
    'eggplant': ['eggplant','eggplants'],
    'greenpepper': ['pepper','peppers','green pepper','green peppers'],
    'pea': ['pea','peas','green pea','green peas'],
    'kiwi': ['kiwi','kiwi fruit','kiwi fruits'],
    'lemon': ['lemon','lemons'],
    'onion': ['onion','onions'],
    'orange': ['orange','oranges'],
    'potatoes': ['potato','potatoes'],
    'bread': ['bread', 'sliced bread'],
    'avocado': ['avocado','avocados'],
    'strawberry': ['strawberry','strawberries'],
    'sweetpotato': ['sweet','sweet potato','sweet potatoes'],
    'tomato': ['tomato','tomatoes'],
    'turnip': ['radish','radishes','white radish','white radishes']
    #'orange02': '/orange02'
}
color_dict = {
    'wh': 'white',
    'br': 'brown',
    'bl': 'blue'
}
number_dict = {
    1: 'one',
    2: 'two',
    3: 'three'
}


def load_i2u(checkpoint_path, **model_params):
    params = checkpoint_path.split("/")[-2].split("_")
    if "gated" in params:
        model = TransformerSentenceLM_FixedImg_gated(**model_params)
    else:
        model = TransformerSentenceLM_FixedImg(**model_params)
    model.pos_encoder = PositionalEncoding(d_model=1024, max_len=152)
    model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    return model

def load_i2u_prev(checkpoint_path, **model_params):
    model = TransformerConditionedLM(**model_params)
    model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    return model

def u2s2t(seq, tacotron2_model, generator, processor, asr_model):
    words = [rev_word_map[ind] for ind in seq if rev_word_map[ind] not in special_words]
    # print(words)
    sequence = np.array(text_to_sequence(' '.join(words), ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    _, mel_outputs_postnet, _, _ = tacotron2_model.inference(sequence)
    with torch.no_grad():
        x = mel_outputs_postnet.squeeze().to(device)
        y_g_hat = generator(mel_outputs_postnet)
        audio = y_g_hat.squeeze()
        
        # audio = audio * 32768.0
        # audio = audio.cpu().numpy().astype('int16')

        audio = audio.cpu().numpy().astype(np.float64)

        input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.float()
        logits = asr_model(input_values.to(device)).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
    return transcription

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

def get_image_info(image_name):
    info=image_name
    name=info.split("_")[0]
    if name=="orange":
        color=info.split("_")[1]
        number=info.split("_")[2]
    else:
        color=info.split("_")[1][0:-1]
        number=info.split("_")[1][-1]
    return name, color, number

def judge_ans(transcription, image_name):
    ans = transcription.split(" ")
    # print(ans)
    right_name = False
    right_color = False
    right_number = False
    right_ans = False

    name, color, number = get_image_info(image_name)

    # 分开两个词 怎么办
    for an in ans:
        if an in name_dict[name]:
            # print("Right name.")
            right_name = True
        if an == color_dict[color]:
            # print("Right color.")
            right_color = True
        if an in number_dict[int(number)]:
            # print("Right number.")
            right_number = True
    if right_name and right_color and right_number:
        right_ans = True
    return right_ans, right_name

def main(model_path=None):

    if model_path is None:
        # model_path = "../../saved_model/I2U/VC_5_captions_224/7*7_img_1024*16*12_99accuracy"
        model_path = "../../saved_model/I2U/VC_5_captions/no_uLM_no_sen_refine_global"
        word_map_path="../../data/processed/VC_5_captions/WORDMAP_coco_5_cap_per_img_1_min_word_freq.json"
    else:
        word_map_path="../../data/processed/SpokenCOCO_LibriSpeech/WORDMAP_coco_1_cap_per_img_1_min_word_freq.json"
    
    with open('../../config.yml') as yml:
        config = yaml.safe_load(yml)
    
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------------------------

    # U2S
    # /net/papilio/storage2/yhaoyuan/LAbyLM/dataprep/RL/image2speech_inference.ipynb

    # tacotron2
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    hparams.max_decoder_steps = config["u2s"]["max_decoder_steps"]
    checkpoint_path = config["u2s"]["tacotron2"]
    tacotron2_model = load_model(hparams)
    tacotron2_model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    tacotron2_model.cuda().eval()

    # --------------------------------------------------------------------------------

    # HiFi-GAN
    # /net/papilio/storage2/yhaoyuan/LAbyLM/dataprep/RL/image2speech_inference.ipynb

    checkpoint_file = config["u2s"]['hifigan']
    config_file = os.path.join(os.path.split(checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
            data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)
    generator = Generator(h).to(device)
    assert os.path.isfile(checkpoint_file)
    checkpoint_dict = torch.load(checkpoint_file, map_location=device)
    generator.load_state_dict(checkpoint_dict['generator'])
    generator.eval()
    generator.remove_weight_norm()

    # --------------------------------------------------------------------------------

    # S2T
    processor = Wav2Vec2Processor.from_pretrained(config["asr"]["model_path"])
    asr_model = Wav2Vec2ForCTC.from_pretrained(config["asr"]["model_path"]).to(device)
    asr_model.eval()

    # --------------------------------------------------------------------------------

    # self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
    # self.imgs = self.h['images']

    # --------------------------------------------------------------------------------

    # Load I2U:

    # model_path = "../../saved_model/I2U/VC_5_captions_224/beam_val_uLM_ungated_no_sen"
    # model_path = "../../saved_model/I2U/VC_5_captions_224/beam_val_uLM_gated_no_sen"
    config_path = glob(model_path + "/config*.yml")[0]
    # config_path = glob(model_path+"/*")
    model_checkpoint = glob(model_path+"/*BEST*.tar")[0]
    # word_map_path="../../data/processed/SpokenCOCO_LibriSpeech/WORDMAP_coco_1_cap_per_img_1_min_word_freq.json"

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
    i2u_model.eval()
    i2u_model.to(device)

    image_resolution = 224

    image_data_path = f"../../data/RL/{str(image_resolution)}"
    # for split in ["TRAIN", "VAL", "TEST"]:
    for split in ["VAL", "TEST"]:
    # for split in ["TRAIN"]:
        imgs, names = load_images(image_data_path, split)
        if is_debug:
            names = names[:10]

        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ans_list = []
        count_list = []
        count = 0
        count_name = 0

        for i in tqdm(range(len(names)), desc=f"Getting {split} Results"):
            img = get_transformed_img(imgs[i], transform)
            # img = get_transformed_img(imgs[i], transform=None)
            img = img.unsqueeze(0)
            name = names[i]
            seqs = i2u_model.decode(image=img, start_unit=word_map["<start>"], end_unit=word_map["<end>"], max_len=150, beam_size=10)
            trans = u2s2t(seq=seqs, tacotron2_model=tacotron2_model, generator=generator, processor=processor, asr_model=asr_model)
            right_ans, right_name =  judge_ans(trans, name)
            ans_list.append(trans)
            count_list.append(right_ans)
            if right_ans:
                count += 1
            if right_name:
                count_name += 1

        with open(model_path + f"/{split}_recognition_results.txt", "w") as f:
            f.write("%-20s\t\t%-20s\n"%("Recognition Accuracy", f"{count/len(names)}"))
            f.write("%-20s\t\t%-20s\n"%("Recog Name Accuracy", f"{count_name/len(names)}")+ "-"*100 +"\n")
            f.write("%-20s\t\t%-50s\t\t%-20s\n"%("Image Name", "Synthesized Answer", "Right Answer?")+ "-"*100+"\n")
            for i in range(len(names)):
                # f.write(f"{names[i]} \t {ans_list[i]} \t {count_list[i]} \n ")
                f.write("%-20s\t\t%-50s\t\t%-20s\n"%(f"{names[i]}", f"{ans_list[i]}", f"{count_list[i]}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str,
                        help='directory to the saved i2u model')
    args = parser.parse_args()
    main(args.model_path)
    # main()