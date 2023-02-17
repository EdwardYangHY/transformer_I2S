from glob import glob
import json
import os
import sys

import numpy as np
from PIL import Image
import resampy
import torch
from torchvision import transforms
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import yaml

sys.path.append('../../egs/U2S/')
from hparams import create_hparams
from train import load_model
from text import text_to_sequence

sys.path.insert(0, "../../egs/dino")
sys.path.insert(0, "../../egs")
from U2U.models import TransformerVAEwithCNN

with open('../../config.yml') as yml:
    config = yaml.safe_load(yml)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# I2U
word_map_path=config["i2u"]["wordmap"]
# Load word map (word2ix)
with open(word_map_path) as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
special_words = {"<unk>", "<start>", "<end>", "<pad>"}

# I2U
sentence_encoder = TransformerVAEwithCNN(len(word_map), config["u2u"]["d_embed"])
sentence_encoder.load_state_dict(torch.load("../../model/U2U/transformer_cnn_29.pt"))
sentence_encoder.eval()
sentence_encoder.to(device)

# U2S
hparams = create_hparams()
hparams.sampling_rate = 22050

# tacotron2
checkpoint_path = config["u2s"]["tacotron2"]
tacotron2_model = load_model(hparams)
tacotron2_model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
tacotron2_model.cuda().eval()

# HiFi-GAN
sys.path.insert(0, '../../egs/hifi-gan')

from models_hifi_gan import Generator
from env import AttrDict

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

config_file = "../../model/UNIVERSAL_V1/config.json"
checkpoint_file = "../../model/UNIVERSAL_V1/g_02500000"

with open(config_file) as f:
    data = f.read()

json_config = json.loads(data)
h = AttrDict(json_config)

generator = Generator(h).to(device)
state_dict_g = load_checkpoint(checkpoint_file, device)
generator.load_state_dict(state_dict_g['generator'])

# S2T
processor = Wav2Vec2Processor.from_pretrained(config["asr"]["path"])
asr_model = Wav2Vec2ForCTC.from_pretrained(config["asr"]["path"]).to(device)


def i2u2s2t(image):
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    seq = sentence_encoder.decode(word_map['<start>'], word_map['<end>'], image=image, beam_size=1)
    words = [rev_word_map[ind] for ind in seq if rev_word_map[ind] not in special_words]
    sequence = np.array(text_to_sequence(' '.join(words), ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    try:
        _, mel_outputs_postnet, _, _ = tacotron2_model.inference(sequence)
        with torch.no_grad():
            y_g_hat = generator(mel_outputs_postnet)
            audio = y_g_hat.squeeze()
        audio = audio.cpu().numpy().astype(np.float64)
        audio = resampy.resample(audio, 22050, 16000)
        # s2t
        input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.float()
        logits = asr_model(input_values.to(device)).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
    except RuntimeError as e:
        transcription = ""
        print(e, flush=True)
    return transcription


def get_transformed_image(img_path):
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = Image.open(img_path)
    image = np.asarray(image)
    image = image.transpose(2, 0, 1)
    image = torch.FloatTensor(image / 255.)
    image = transform(image)
    image = image.numpy()
    return image


obj2color = {
    'r': 'RED',
    'g': 'GREEN',
    'b': 'BLUE',
}


zero_shot_list = [('onion', 'r'), ('orange', 'r'), ('sweet_potato', 'g'), ('hyacinth_bean', 'g'), ('white_radish', 'b'), ('green_pepper', 'b')]
for food_name, c in zero_shot_list:
    img_paths = glob(f"../../data/I2U/image_color/{food_name}/*/*_{c}.jpg")
    assert len(img_paths) == 120

    count = 0
    for img_path in img_paths:
        image = get_transformed_image(img_path)
        transcript = i2u2s2t(image)
        target = obj2color[c] + " " + food_name.upper().replace("_", " ")
        if target in transcript:
            count += 1
    print(food_name, c, count, flush=True)