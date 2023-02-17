import json
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import resampy


sys.path.append('../../egs/U2S/')
sys.path.append('../../egs/U2S/waveglow')
from hparams import create_hparams
from train import load_model
from text import text_to_sequence
from waveglow.denoiser import Denoiser

sys.path.insert(0, "../../egs/dino")
sys.path.insert(0, "../../egs")
from U2U.models import TransformerVAEwithViT
from U2U.dataset import UnitImageDataset

with open('../../config.yml', 'r') as yml:
    config = yaml.safe_load(yml)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# I2U
word_map_path=config["i2u"]["wordmap"]
# Load word map (word2ix)
with open(word_map_path, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
special_words = {"<unk>", "<start>", "<end>", "<pad>"}

# I2U
sentence_encoder = TransformerVAEwithViT(len(word_map), config["u2u"]["d_embed"])
sentence_encoder.load_state_dict(torch.load(config["u2u"]["path"]))
sentence_encoder.eval()
sentence_encoder.to(device)

# U2S
hparams = create_hparams()
hparams.sampling_rate = 22050

# tacotron2
checkpoint_path = config["u2s"]["tacotron2"]
tacotron2_model = load_model(hparams)
tacotron2_model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
tacotron2_model.cuda().eval().half()

# WaveGlow
waveglow_path = config["u2s"]["waveglow"]
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)

# S2T
processor = Wav2Vec2Processor.from_pretrained(config["asr"]["path"])
asr_model = Wav2Vec2ForCTC.from_pretrained(config["asr"]["path"]).to(device)


def i2u2s2t(x, padding_mask, image):
    seq = sentence_encoder.beam_decode(x, padding_mask, image, word_map['<start>'], word_map['<end>'])
    words = [rev_word_map[ind] for ind in seq if rev_word_map[ind] not in special_words]
    sequence = np.array(text_to_sequence(' '.join(words), ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    try:
        _, mel_outputs_postnet, _, _ = tacotron2_model.inference(sequence)
        with torch.no_grad():
            audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
        audio = audio.squeeze()
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


obj2color = {
    'apple': 'red',
    'banana': 'yellow',
    'carrot': 'orange',
    'cherry': 'black',
    'cucumber': 'green',
    'egg': 'chicken',
    'eggplant': 'purple',
    'green_pepper': 'green',
    'hyacinth_bean': 'green',
    'kiwi_fruit': 'brown',
    'lemon': 'yellow',
    'onion': 'yellow',
    'orange': 'orange',
    'potato': 'brown',
    'sliced_bread': 'yellow',
    'small_cabbage': 'green',
    'strawberry': 'red',
    'sweet_potato': 'brown',
    'tomato': 'red',
    'white_radish': 'white',
}

def add_indications(name, idx):
    preposition = 'an' if name[0] in ['a', 'o', 'e'] else 'a'
    color = obj2color[name]
    preposition2 = 'an' if color[0] in ['a', 'o', 'e'] else 'a'
    ll = [name, f'{preposition} {name}',
          f'{preposition2} {color} {name}', f"i want {preposition} {name}"]
    return ll[idx].replace("_", " ").upper()

def load_image(img_path):
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = Image.open(img_path)
    image = np.asarray(image)
    image = image.transpose(2, 0, 1)
    image = torch.FloatTensor(image / 255.)
    image = transform(image)
    image = image.numpy()
    return image

res_dict = dict()
for i, obj in enumerate(obj2color):
    for n in range(10):
        res_dict[10*i + n] = dict()
        for k in range(4):
            res_dict[10*i + n][add_indications(obj, k)] = 0

print(res_dict, flush=True)

with open('../../config.yml') as yml:
    config = yaml.safe_load(yml)

loader = DataLoader(
    UnitImageDataset('../../data/I2U/processed/', f'coco_{str(config["i2u"]["captions_per_image"])}_cap_per_img_{str(config["i2u"]["min_word_freq"])}_min_word_freq', 'TEST'),
    batch_size=1,
    shuffle=False,
    num_workers=config["u2u"]["num_workers"],
    pin_memory=True,
    )

for epoch in range(100):
    for i, (units, padding_masks, seq_lens, imgs) in enumerate(loader):
        units = units.cuda()
        padding_masks = padding_masks.cuda()
        imgs = imgs.cuda()

        transcription = i2u2s2t(units, padding_masks, imgs)
        if transcription in res_dict[i//4]:
            res_dict[i//4][transcription] += 1
    print(epoch, res_dict, flush=True)