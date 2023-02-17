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

from custom_policy2 import CustomTD3PolicyCNN, CustomDDPG

sys.path.append('../../egs/U2S/')
from hparams import create_hparams
from train import load_model
from text import text_to_sequence

sys.path.insert(0, "../../egs/dino")
sys.path.insert(0, "../../egs")
from U2U.models import TransformerVAEwithCNN

# config path需要更改
with open('../../config.yml') as yml:
    config = yaml.safe_load(yml)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------------

# I2U
# TODO： Wordmap 位置需要更改
word_map_path=config["i2u"]["wordmap"]
# Load word map (word2ix)
with open(word_map_path) as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
special_words = {"<unk>", "<start>", "<end>", "<pad>"}

# I2U
# TODO: 模型位置需要更改
sentence_encoder = TransformerVAEwithCNN(len(word_map), config["u2u"]["d_embed"])
sentence_encoder.load_state_dict(torch.load(config["u2u"]["path2"]))
sentence_encoder.eval()
sentence_encoder.to(device)

# --------------------------------------------------------------------------------

# U2S
# TODO: 调用匹配的tacotron2 和 HifiGAN 具体参考
# /net/papilio/storage2/yhaoyuan/LAbyLM/dataprep/RL/image2speech_inference.ipynb
hparams = create_hparams()
hparams.sampling_rate = 22050

# tacotron2
checkpoint_path = config["u2s"]["tacotron2"]
tacotron2_model = load_model(hparams)
tacotron2_model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
tacotron2_model.cuda().eval()

# --------------------------------------------------------------------------------

# HiFi-GAN
# TODO: 调用匹配的tacotron2 和 HifiGAN 具体参考
# /net/papilio/storage2/yhaoyuan/LAbyLM/dataprep/RL/image2speech_inference.ipynb
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

# --------------------------------------------------------------------------------

# S2T
# TODO: 调用对应的， tune的ASR
processor = Wav2Vec2Processor.from_pretrained(config["asr"]["path"])
asr_model = Wav2Vec2ForCTC.from_pretrained(config["asr"]["path"]).to(device)

# --------------------------------------------------------------------------------

# 同样的模型结构应该不需要调整
# TODO: 测试： image + sentence embedding 能不能解码出 seq。

def i2u2s2t(action):
    action = torch.from_numpy(action).unsqueeze(0).to(device)
    seq = sentence_encoder.decode(word_map['<start>'], word_map['<end>'], action=action, beam_size=5)
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

# --------------------------------------------------------------------------------

class SpoLacq(gym.Env):
    def __init__(
        self,
        d_embed: int,
        d_img_features: int,
        img_list: list,
        rewards: list = [1, 0, 0],
    ):
        super().__init__()
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(d_img_features+d_embed+2,))
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
        
        self.rewards = rewards
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
        transcription = i2u2s2t(action[:-2])
        alpha = action[-2:]

        # uses the last 2 action[-2:] to predict the pos of the right ans
        pred_num = np.argmax(alpha)
        if ans_num == pred_num:
            self.rl_rewards.append(1)
        else:
            self.rl_rewards.append(0)

        # TODO: 改变 transcription的judge方式，参考：
        # /net/papilio/storage2/yhaoyuan/LAbyLM/dataprep/RL/image2speech_inference.ipynb
        if transcription[:7] == "I WANT ":
            self.utt_rewards.append(1)
        else:
            self.utt_rewards.append(0)

        img_path = self.img_list[ans_num]
        transcription_ans = img_path.split("/")[5].replace("_", " ").upper()
        preposition = 'an' if transcription_ans[0] in ['A', 'O', 'E'] else 'a'
        print(
            self.num_step,
            self.img_list[self.data_num1].split("/")[5],
            self.img_list[self.data_num2].split("/")[5],
            f"ANSWER: {transcription_ans}",
            flush=True,
            )
        if transcription_ans in transcription:
            if f"i want {preposition} {transcription_ans}".upper() == transcription:
                print(self.rewards[0], self.num_step, transcription, flush=True)
                return self.rewards[0]
            else:
                print(self.rewards[1], self.num_step, transcription, flush=True)
                return self.rewards[1]
        else:
            print(self.rewards[2], self.num_step, transcription, flush=True)
            return self.rewards[2]
        
        # ---------------------------
    
    def save_rewards(self, path):
        np.save(path, np.array(self.rl_rewards))


def test(env, model, num_episode: int = 1000) -> None:
    """Test the learnt agent."""
    
    total_reward = 0
    for i in range(num_episode):
        state = env.reset()
        
        # Agent gets an environment state and returns a decided action
        action, _ = model.predict(state, deterministic=True)
        
        # Environment gets an action from the agent, proceeds the time step,
        # and returns the new state and reward etc.
        state, reward, done, info = env.step(action)
        total_reward += reward
    print(f"total_reward: {total_reward}\n", flush=True)


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    action_noise = NormalActionNoise(
        mean=np.zeros(49*2048+config["u2u"]["d_embed"]+2),
        sigma=np.concatenate([np.zeros(49*2048), 0.3*np.ones(config["u2u"]["d_embed"]), np.zeros(2)]),
        )
    
    # --------------------------------------------------------------------------------
    # TODO: change the file lists by new divided json files.
    # see /net/papilio/storage2/yhaoyuan/transformer_I2S/data/food_dataset_VC_shuffle.json
    foods_incremental = [
        'lemon',
        'onion',
        'orange',
        'potato',
        'sliced_bread',
        'small_cabbage',
        'strawberry',
        'sweet_potato',
        'tomato',
        'white_radish',
    ]

    img_list_train = glob('../../data/I2U/image/*/train_number*/*.jpg')
    img_list_eval = glob('../../data/I2U/image/*/test_number[12]/*.jpg')
    img_list_test = glob('../../data/I2U/image/*/test_number3/*.jpg')

    if config["rl"]["incremental"]:
        img_list_train = [path for path in img_list_train if path.split("/")[5] not in foods_incremental]
        img_list_eval = [path for path in img_list_eval if path.split("/")[5] not in foods_incremental]
        img_list_test = [path for path in img_list_test if path.split("/")[5] not in foods_incremental]

    # --------------------------------------------------------------------------------

    env = SpoLacq(
        d_embed=config["u2u"]["d_embed"],
        d_img_features=49*2048,
        img_list=img_list_train,
        rewards=[1, 0, 0],
        )
    
    eval_env = SpoLacq(
        d_embed=config["u2u"]["d_embed"],
        d_img_features=49*2048,
        img_list=img_list_eval,
        rewards=[1, 0, 0],
        )
    
    test_env = SpoLacq(
        d_embed=config["u2u"]["d_embed"],
        d_img_features=49*2048,
        img_list=img_list_test,
        rewards=[1, 0, 0],
        )

    model2 = CustomDDPG(
        CustomTD3PolicyCNN,
        env,
        learning_rate=config["rl"]["learning_rate"],
        buffer_size=config["rl"]["buffer_size"],
        learning_starts=config["rl"]["learning_starts"],
        batch_size=config["rl"]["batch_size"],
        train_freq=4,
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
    # model2.load_param_from_ddpg(CustomDDPG.load("./logs_ddpg_without_embed_icassp/best_model.zip"))
    # model2.fix_param()
    model2.use_embed()

    model2.learn(
        total_timesteps=100000,
        eval_env=eval_env,
        eval_freq=1000,
        n_eval_episodes=config["rl"]["n_eval_episodes_ddpg"],
        eval_log_path="./logs_ddpg_embed_icassp/",
        )
    
    eval_env.save_rewards("./logs_ddpg_embed_icassp/rl_accuracy.npy")