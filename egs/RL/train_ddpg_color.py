import datetime
from glob import glob
import json
import os
import sys

import gym
import numpy as np
from PIL import Image
import resampy
from stable_baselines3.common.noise import NormalActionNoise
import torch
from torchvision import transforms
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import yaml

from custom_policy import CustomTD3PolicyCNN, CustomDDPG

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


def i2u2s2t(action):
    action = torch.from_numpy(action).unsqueeze(0).to(device)
    seq = sentence_encoder.decode(word_map['<start>'], word_map['<end>'], action=action)
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


class SpoLacq(gym.Env):
    def __init__(
        self,
        d_embed: int,
        d_img_features: int,
        img_list: list,
    ):
        super().__init__()
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(d_img_features+d_embed,))
        self.observation_space = gym.spaces.Dict(
            dict(
                state=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                leftimage=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3, 224, 224), dtype=np.float64),
                rightimage=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3, 224, 224), dtype=np.float64),
            )
        )
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        self.img_list = img_list
        self.make_food_storage(img_list)
        
        self.utt_rewards = list()
        self.rl_rewards = list()
        
        self.num_step = 0
        self.reset()
    
    def make_food_storage(self, img_list: list):
        self.food_storage = list()
        for img_path in img_list:
            image = Image.open(img_path)
            image = np.asarray(image)
            self.food_storage.append(image)
    
    def get_transformed_image(self, data_num):
        transform = transforms.Normalize(mean=self.mean, std=self.std)
        image = self.food_storage[data_num]
        image = image.transpose(2, 0, 1)
        image = torch.FloatTensor(image / 255.)
        image = transform(image)
        image = image.numpy()
        return image

    def reset(self):
        self.data_num1 = np.random.randint(len(self.img_list))
        self.data_num2 = np.random.randint(len(self.img_list))
        self.internal_state = np.random.randint(0, 256, size=3)
        data1 = self.get_transformed_image(self.data_num1)
        data2 = self.get_transformed_image(self.data_num2)
        state = dict(
            state=(self.internal_state/255.-self.mean) / self.std,
            leftimage=data1,
            rightimage=data2,
        )
        return state

    def step(self, action):
        reward = self.reward(action)
        self.data_num1 = np.random.randint(len(self.img_list))
        self.data_num2 = np.random.randint(len(self.img_list))
        self.internal_state = np.random.randint(0, 256, size=3)
        data1 = self.get_transformed_image(self.data_num1)
        data2 = self.get_transformed_image(self.data_num2)
        state = dict(
            state=(self.internal_state/255.-self.mean) / self.std,
            leftimage=data1,
            rightimage=data2,
        )
        return state, reward, True, {}
    
    def get_ans(self):
        rgb_1 = np.mean(self.food_storage[self.data_num1], axis=(0,1))
        rgb_2 = np.mean(self.food_storage[self.data_num2], axis=(0,1))
        distance_1 = np.linalg.norm(rgb_1-self.internal_state)
        distance_2 = np.linalg.norm(rgb_2-self.internal_state)
        return 0 if distance_1 < distance_2 else 1

    def reward(self, action):
        self.num_step += 1
        ans = self.get_ans()
        ans_num = self.data_num1 if ans == 0 else self.data_num2
        transcription = i2u2s2t(action)

        img_path = self.img_list[ans_num]
        food_ans = img_path.split("/")[5].replace("_", " ").upper()
        
        if img_path[-5] == "r":
            color_ans = "RED"
        elif img_path[-5] == "g":
            color_ans = "GREEN"
        elif img_path[-5] == "b":
            color_ans = "BLUE"
        else:
            raise ValueError("undefined colr")
        
        transcription_ans = f"{color_ans} {food_ans}"
        
        print(
            self.num_step,
            self.img_list[self.data_num1].split("/")[5],
            self.img_list[self.data_num2].split("/")[5],
            f"ANSWER: {transcription_ans}",
            flush=True,
            )
        if transcription_ans in transcription:
            print(1, self.num_step, transcription, flush=True)
            return 1
        else:
            print(0, self.num_step, transcription, flush=True)
            return 0
    
    def save_rewards(self, log_dir, data_name: str):
        log_dir = log_dir if log_dir[-1] == "/" else log_dir + "/"
        np.save(f"{log_dir}utt_rewards_{data_name}.npy", np.array(self.utt_rewards))
        np.save(f"{log_dir}rl_rewards_{data_name}.npy", np.array(self.rl_rewards))


if __name__ == "__main__":
    action_noise = NormalActionNoise(
        mean=np.zeros(7*7*2048+config["u2u"]["d_embed"]),
        sigma=np.concatenate([np.zeros(7*7*2048), 0.3*np.ones(config["u2u"]["d_embed"])]),
        )

    img_list_train = glob('../../data/I2U/image_color/*/train_number*/*.jpg')
    img_list_eval = glob('../../data/I2U/image_color/*/test_number[12]/*.jpg')
    img_list_test = glob('../../data/I2U/image_color/*/test_number3/*.jpg')

    env = SpoLacq(
        d_embed=config["u2u"]["d_embed"],
        d_img_features=49*2048,
        img_list=img_list_train,
        )
    
    eval_env = SpoLacq(
        d_embed=config["u2u"]["d_embed"],
        d_img_features=49*2048,
        img_list=img_list_eval,
        )
    
    test_env = SpoLacq(
        d_embed=config["u2u"]["d_embed"],
        d_img_features=49*2048,
        img_list=img_list_test,
        )

    model2 = CustomDDPG(
        CustomTD3PolicyCNN,
        env,
        learning_rate=config["rl"]["learning_rate"],
        buffer_size=config["rl"]["buffer_size"],
        learning_starts=config["rl"]["learning_starts"],
        batch_size=config["rl"]["batch_size"],
        action_noise=action_noise,
        replay_buffer_kwargs = dict(handle_timeout_termination=False),
        tensorboard_log='./spolacq_tmplog/',
        policy_kwargs=dict(
            net_arch=dict(
                pi=[[150, 75, 2], [150, 75, config["u2u"]["d_embed"]]],
                qf=[150+50+config["u2u"]["d_embed"], (150+50+config["u2u"]["d_embed"])//2, 1],
                )
            )
        )
    model2.use_embed()

    model2.learn(
        total_timesteps=50000,
        eval_env=eval_env,
        eval_freq=200,
        n_eval_episodes=config["rl"]["n_eval_episodes_ddpg"],
        eval_log_path="./logs_ddpg_embed_icassp/",
        )