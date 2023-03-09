import datetime
from glob import glob
import json
import os
import sys

import gym
import numpy as np
from PIL import Image
from imageio import imread
import resampy
from stable_baselines3.common.noise import NormalActionNoise
import torch
from torchvision import transforms
import yaml
from tqdm import tqdm
from custom_policy_1_3 import CustomTD3PolicyCNN, CustomDDPG, CustomTD3PolicyCNN_new
import model_loader as loader
from RL_envs import SpoLacq, SpoLacq_new

# config path需要更改
# --------------------------------------------------------------------------------
# Params, models
with open('../../config.yml') as yml:
    config = yaml.safe_load(yml)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_debug = True if sys.gettrace() else False

word_map_path="../../data/processed/SpokenCOCO_LibriSpeech/WORDMAP_coco_1_cap_per_img_1_min_word_freq.json"
with open(word_map_path) as j:
    word_map = json.load(j)

i2u_model_path = "../../saved_model/I2U/VC_5_captions_224/beam_val_no_uLM_no_sen"
train_ID = i2u_model_path.split("/")[-1]
U2S_checkpoint = config["u2s"]["tacotron2"]
vocoder_checkpoint = config["u2s"]['hifigan']
ASR_checkpoint = config["asr"]["model_path"]


# --------------------------------------------------------------------------------
# I2U
print("Load I2U model")

config["u2u"] = {}
config["u2u"]["d_embed"] = 0

i2u_model = loader.load_I2U(
    model_path=i2u_model_path, 
    word_map=word_map, 
    device=device
    )
print("Load I2U model complete.")

# --------------------------------------------------------------------------------

# U2S, HIFI-GAN refer to:
# /net/papilio/storage2/yhaoyuan/LAbyLM/dataprep/RL/image2speech_inference.ipynb

print("Load U2S model")
u2s_model = loader.load_U2S(checkpoint_path=U2S_checkpoint)
vocoder_generator = loader.load_vocoder(checkpoint_path=vocoder_checkpoint, device=device)
print("Load U2S model complete.")

# --------------------------------------------------------------------------------
print("Load ASR model")
# S2T
asr_processor, asr_model = loader.load_ASR(checkpoint_path=ASR_checkpoint, device=device)
print("Load ASR model complete.")
# --------------------------------------------------------------------------------

# 同样的模型结构应该不需要调整
# TODO: 测试： image + sentence embedding 能不能解码出 seq。
# 可以解码

mouth = loader.Unit2Text(
    i2u_model=i2u_model,
    u2s_model=u2s_model,
    vocoder_generator=vocoder_generator,
    asr_processor=asr_processor,
    asr_model=asr_model,
    word_map=word_map,
    device=device,
)

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


    # img_data_path = "/net/papilio/storage2/yhaoyuan/transformer_I2S/data/food_dataset_VC_shuffle.json"
    # with open(img_data_path, "r") as f:
    #     img_data = json.load(f)
    # img_list_train = [img_data["image_base_path"] + pairdata["image"] for pairdata in img_data["data"]["train"]]
    # img_list_eval = [img_data["image_base_path"] + pairdata["image"] for pairdata in img_data["data"]["val"]]
    # img_list_test = [img_data["image_base_path"] + pairdata["image"] for pairdata in img_data["data"]["test"]]
    
    # if is_debug:
    #     img_list_train = img_list_train[:10]
    #     img_list_eval = img_list_eval[:1]
    #     img_list_test = img_list_test[:1]


    img_hdf5_train = "../../data/RL/TRAIN_food_dataset_VC_IMAGES.hdf5"
    img_list_train = "../../data/RL/TRAIN_food_dataset_VC_NAMES.json"
    img_hdf5_eval = "../../data/RL/VAL_food_dataset_VC_IMAGES.hdf5"
    img_list_eval = "../../data/RL/VAL_food_dataset_VC_NAMES.json"
    img_hdf5_test= "../../data/RL/TEST_food_dataset_VC_IMAGES.hdf5"
    img_list_test = "../../data/RL/TEST_food_dataset_VC_NAMES.json"
    with open(img_list_train, "r") as f:
        img_list_train = json.load(f)
    with open(img_list_eval, "r") as f:
        img_list_eval = json.load(f)
    with open(img_list_test, "r") as f:
        img_list_eval = json.load(f)

    # img_list_train = glob('../../data/image_synthesize/*/train_number*/*.jpg')
    # img_list_eval = glob('../../data/image_synthesize/*/test_number[12]/*.jpg')
    # img_list_test = glob('../../data/image_synthesize/*/train_number*/*.jpg')

    # if is_debug:
    #     img_train_list = img_list_train[:10]
    #     img_eval_list = img_list_train[:1]

    # --------------------------------------------------------------------------------
    print("Prepare enviroment")
    # NOTE: changed reward
    env = SpoLacq_new(
        d_embed=config["u2u"]["d_embed"],
        d_img_features=49*2048,
        img_list=img_list_train,
        img_hdf5=img_hdf5_train,
        i2u2s2t=mouth.i2u2s2t,
        rewards=[1, 0.1, 0],
        )
    
    eval_env = SpoLacq_new(
        d_embed=config["u2u"]["d_embed"],
        d_img_features=49*2048,
        img_list=img_list_eval,
        img_hdf5=img_hdf5_eval,
        i2u2s2t=mouth.i2u2s2t,
        rewards=[1, 0, 0],
        )
    
    
    # test_env = SpoLacq(
    #     d_embed=config["u2u"]["d_embed"],
    #     d_img_features=49*2048,
    #     img_list=img_list_test,
    #     img_hdf5=img_hdf5_test,
    #     i2u2s2t=mouth.i2u2s2t,
    #     rewards=[1, 0, 0],
    #     )
    
    features_dim = 2048
    print("Prepare enviroment complete.")
    print("Set up agent.")
    model2 = CustomDDPG(
        # CustomTD3PolicyCNN,
        CustomTD3PolicyCNN_new,
        env,
        learning_rate=config["rl"]["learning_rate"],
        buffer_size=config["rl"]["buffer_size"],
        learning_starts=config["rl"]["learning_starts"],
        batch_size=config["rl"]["batch_size"],
        train_freq=4,
        action_noise=action_noise,
        replay_buffer_kwargs = dict(handle_timeout_termination=False),
        tensorboard_log=f'./spolacq_tmplog_{train_ID}/',
        policy_kwargs=dict(
            net_arch=dict(
                pi=[[3*features_dim, 3*features_dim, 2], [3*features_dim, 3*features_dim, config["u2u"]["d_embed"]]],
                qf=[4*features_dim+config["u2u"]["d_embed"], (4*features_dim+config["u2u"]["d_embed"]), (4*features_dim+config["u2u"]["d_embed"]), 1],
                # qf=[4*features_dim+config["u2u"]["d_embed"], (4*features_dim+config["u2u"]["d_embed"])//2, 1],
                ),
            features_extractor_kwargs=dict(features_dim=features_dim),
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs=dict(weight_decay=1.0),
            )
        )
    # model2.load_param_from_ddpg(CustomDDPG.load("./logs_ddpg_without_embed_icassp/best_model.zip"))
    # model2.fix_param()
    model2.use_embed()
    print("Set up agent complete.")

    print("Start Learning.")
    model2.learn(
        total_timesteps=50000,
        # total_timesteps=100,
        #'''Removed in version 1.7.0'''
        eval_env=eval_env,
        eval_freq=1000,
        n_eval_episodes=config["rl"]["n_eval_episodes_ddpg"],
        eval_log_path=f"./logs_ddpg_{train_ID}/",
        )
    
    env.save_rewards(f"./spolacq_tmplog_{train_ID}/rl_accuracy.npy")