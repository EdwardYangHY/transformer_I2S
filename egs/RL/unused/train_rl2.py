import datetime
from glob import glob
import sys
import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import yaml
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.noise import NormalActionNoise

from my_custom_policy2 import CustomTD3Policy2, CustomDDPG

sys.path.append('../../dataprep/RL')
from i2u2s2t5 import i2u2s2t


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
        
        self.rewards = list()
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
        transcription_ans = img_path.split("/")[5].replace("_", " ").upper()
        print(
            self.num_step,
            self.img_list[self.data_num1].split("/")[5],
            self.img_list[self.data_num2].split("/")[5],
            f"ANSWER: {transcription_ans}",
            flush=True,
            )
        if transcription_ans in transcription:
            self.rl_rewards.append(1)
            if "I WANT " in transcription:
                self.rewards.append(1)
                self.utt_rewards.append(1)
                print(1, self.num_step, transcription, flush=True)
                return 1
            else:
                self.rewards.append(0)
                self.utt_rewards.append(0)
                print(0, self.num_step, transcription, flush=True)
                return 0
        else:
            self.rl_rewards.append(0)
            if "I WANT " in transcription:
                self.rewards.append(0)
                self.utt_rewards.append(1)
                print(0, self.num_step, transcription, flush=True)
                return 0
            else:
                self.rewards.append(0)
                self.utt_rewards.append(0)
                print(0, self.num_step, transcription, flush=True)
                return 0
    
    def save_rewards(self, data_name: str):
        np.save(f"../../model/RL/rewards_{data_name}.npy", np.array(self.rewards))
        np.save(f"../../model/RL/utt_rewards_{data_name}.npy", np.array(self.utt_rewards))
        np.save(f"../../model/RL/rl_rewards_{data_name}.npy", np.array(self.rl_rewards))


if __name__ == "__main__":
    with open('../../config.yml', 'r') as yml:
        config = yaml.safe_load(yml)

    action_noise = NormalActionNoise(
        mean=np.zeros(7*7*2048+config["u2u"]["d_embed"]),
        sigma=np.concatenate([np.zeros(7*7*2048), np.ones(config["u2u"]["d_embed"])]),
        )

    env = SpoLacq(
        d_embed=config["u2u"]["d_embed"],
        d_img_features=7*7*2048,
        img_list=glob('../../data/I2U/image/*/train_number*/*.jpg'),
        )
    
    eval_env = SpoLacq(
        d_embed=config["u2u"]["d_embed"],
        d_img_features=7*7*2048,
        img_list=glob('../../data/I2U/image/*/test_number[12]/*.jpg'),
        )
    
    model  = DQN.load("./logs_dqn/best_model")
    # model  = CustomDDPG.load(config["rl"]["model"])
    model2 = CustomDDPG(
        CustomTD3Policy2,
        env,
        action_noise=action_noise,
        buffer_size=10000,  # Require 52 GB RAM
        replay_buffer_kwargs = dict(handle_timeout_termination=False),
        tensorboard_log='./spolacq_tmplog/',
        )
    model2.load_param_from_dqn(model)
    model2.fix_param()
    model2.learn(
        total_timesteps=100000,
        eval_env=eval_env,
        eval_freq=1000,
        n_eval_episodes=100,
        eval_log_path="./logs/",
        )
    
    eval_env.save_rewards(data_name=datetime.datetime.now().strftime('%y%m%d%H%M')+"_lstm")