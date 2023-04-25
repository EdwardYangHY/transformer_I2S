import datetime
from glob import glob
import json
import os
import sys

import gym
import numpy as np
import h5py
from PIL import Image
from imageio import imread
import resampy
from stable_baselines3.common.noise import NormalActionNoise
import torch
from torchvision import transforms
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import yaml
from tqdm import tqdm
from custom_policy_1_3 import CustomTD3PolicyCNN, CustomDDPG, CustomTD3PolicyCNN_new

sys.path.append("../I2U")
from utils_synthesize import *
from judge_asr import judge_ans, get_image_info

# sys.path.append("../..")
# import hifigan
# from hifigan.env import AttrDict
# from hifigan.models import Generator

# sys.path.append("../U2S")
# from hparams import create_hparams
# from train import load_model
# from text import text_to_sequence


# sys.path.append("../I2U")
# from models import TransformerSentenceLM
# from models_k import TransformerVAEwithCNN
# from models_modified import TransformerSentenceLM_FixedImg

# config path需要更改
with open('../../config.yml') as yml:
    config = yaml.safe_load(yml)

if 'u2u' not in config.keys():
    config['u2u'] = {}
    config['u2u']['d_embed'] = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_debug = True if sys.gettrace() else False

# --------------------------------------------------------------------------------
print("Load u2s2t model")
# tacotron_max_decoder_step = config["u2s"]["max_decoder_steps"]
# tacotron_checkpoint_path = config["u2s"]["tacotron2"]
# hifigan_checkpoint_path = config["u2s"]['hifigan']

tacotron_max_decoder_step = 500
tacotron_checkpoint_path = "../../saved_model/U2S/ourdir_mapping_warmstart_from_gtts/checkpoint_47000"
hifigan_checkpoint_path = "../../hifigan/FOOD_V1_24K_Speaker3/generator_v1_24k"
asr_checkpoint_path = config["asr"]["model_path"]

# tacotron_model = load_tacotron2(tacotron_checkpoint_path, tacotron_max_decoder_step)

tacotron_model = load_tacotron2(
        tacotron_checkpoint_path, 
        max_decoder_step=tacotron_max_decoder_step,
        sr=24000,
        vocab_size=1024
    )
hifigan_model = load_hifigan(hifigan_checkpoint_path, device)
asr_model, asr_processor = load_asr(asr_checkpoint_path, device)
# --------------------------------------------------------------------------------
print("Load i2u model")
i2u_checkpoint = "../../saved_model/I2U/origin_5_captions_256/baseline_lr-3_no_LM/bleu-4_BEST_checkpoint_coco_5_cap_per_img_1_min_word_freq_gpu.pth.tar"
i2u_config_path = "../../saved_model/I2U/origin_5_captions_256/baseline_lr-3_no_LM/config_sentence.yml"
word_map_path = "../../data/processed/origin_5_captions_256/WORDMAP_coco_5_cap_per_img_1_min_word_freq.json"

with open(word_map_path) as f:
    word_map = json.load(f)
rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
special_words = {"<unk>", "<start>", "<end>", "<pad>"}

i2u_model = load_i2u(i2u_checkpoint, i2u_config_path, len(word_map))
i2u_model.eval()
i2u_model.to(device)
# --------------------------------------------------------------------------------

def i2u2s2t(action):
    action = torch.from_numpy(action).unsqueeze(0).to(device)
    seqs = i2u_model.decode(action=action, start_unit=word_map["<start>"], end_unit=word_map["<end>"], max_len=150, beam_size=10)
        
    words = seq2words(seq=seqs, rev_word_map=rev_word_map, special_words=special_words)
    # audio = u2s(
    #     words=words,
    #     tacotron2_model=tacotron_model,
    #     hifigan_model=hifigan_model,
    #     device=device
    #     )
    audio = u2s(
        words=words,
        tacotron2_model=tacotron_model,
        hifigan_model=hifigan_model,
        device=device
        )
    
    trans = s2t(audio=audio, asr_model=asr_model, asr_processor=asr_processor, device=device)
    return trans

def i2u2s2t_img(transformed_img):
    img = torch.FloatTensor(transformed_img).unsqueeze(0).to(device)
    seqs = i2u_model.decode(image=img, start_unit=word_map["<start>"], end_unit=word_map["<end>"], max_len=150, beam_size=10)
     
    words = seq2words(seq=seqs, rev_word_map=rev_word_map, special_words=special_words)
    # audio = u2s(
    #     words=words,
    #     tacotron2_model=tacotron_model,
    #     hifigan_model=hifigan_model,
    #     device=device
    #     )
    audio = u2s(
        words=words,
        tacotron2_model=tacotron_model,
        hifigan_model=hifigan_model,
        device=device
        )
    
    trans = s2t(audio=audio, asr_model=asr_model, asr_processor=asr_processor, device=device)
    return trans

# --------------------------------------------------------------------------------

def load_images(data_path, split):
    image_hdf5 = glob(data_path+f"/{split}*.hdf5")[0]
    image_names = glob(data_path+f"/{split}*.json")[0]
    h = h5py.File(image_hdf5, 'r')
    images = h['images']
    with open(image_names, "r") as f:
        names = json.load(f)
    # imgs = []
    # for i in range(len(names)):
    #     imgs.append(images[i])
    return images, names

class SpoLacq(gym.Env):
    def __init__(
        self,
        d_embed: int,
        d_img_features: int,
        img_list: list = None,
        img_hdf5: str = None,
        img_split: str = None,
        rewards: list = [1, 0, 0],
    ):
        super().__init__()
        # self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(d_img_features+d_embed+2,))
        # self.observation_space = gym.spaces.Dict(
        #     dict(
        #         state=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
        #         leftimage=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3, 224, 224), dtype=np.float64),
        #         rightimage=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3, 224, 224), dtype=np.float64),
        #     )
        # )

        # Version 1.7.0 don't take inf as the upper/lower bound
        self.action_space = gym.spaces.Box(low=-100, high=100, shape=(d_img_features+d_embed+2,))
        self.observation_space = gym.spaces.Dict(
            dict(
                state=gym.spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float64),
                # leftimage=gym.spaces.Box(low=-100, high=100, shape=(3, 224, 224), dtype=np.float64),
                # rightimage=gym.spaces.Box(low=-100, high=100, shape=(3, 224, 224), dtype=np.float64),
                leftimage=gym.spaces.Box(low=-100, high=100, shape=(3, 256, 256), dtype=np.float64),
                rightimage=gym.spaces.Box(low=-100, high=100, shape=(3, 256, 256), dtype=np.float64),
            )
        )
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transform = transforms.Normalize(mean=self.mean, std=self.std)
        
        if img_hdf5 is None:
            self.img_list = [img.split("/")[-1] for img in img_list]
            self.make_food_storage(img_list)
        else:
            imgs, names = load_images(img_hdf5, img_split)
            self.food_storage = list()
            for i in range(len(names)):
                self.food_storage.append(imgs[i])
            self.img_list = names
        
        self.rewards = rewards
        self.utt_rewards = list()
        self.rl_rewards = list()
        
        self.num_step = 0
        self.reset()
    
    def make_food_storage(self, img_list: list):
        self.food_storage = list()
        for img_path in tqdm(img_list):
            img = imread(img_path)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2)
            # img = imresize(img, (256, 256))
            resolution = 224
            image = np.array(Image.fromarray(img).resize((resolution, resolution)))

            # image = Image.open(img_path)
            # image = np.asarray(image)
            self.food_storage.append(image)
    
    def get_transformed_image(self, data_num):
        # transform = transforms.Normalize(mean=self.mean, std=self.std)
        image = self.food_storage[data_num]
        # image = image.transpose(2, 0, 1)
        image = torch.FloatTensor(image / 255.)
        image = self.transform(image)
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
        # rgb_1 = np.mean(self.food_storage[self.data_num1], axis=(0,1))
        # rgb_2 = np.mean(self.food_storage[self.data_num2], axis=(0,1))
        rgb_1 = np.mean(self.food_storage[self.data_num1], axis=(1,2))
        rgb_2 = np.mean(self.food_storage[self.data_num2], axis=(1,2))
        distance_1 = np.linalg.norm(rgb_1-self.internal_state)
        distance_2 = np.linalg.norm(rgb_2-self.internal_state)
        return 0 if distance_1 < distance_2 else 1

    def reward(self, action):
        self.num_step += 1
        ans = self.get_ans()
        ans_num = self.data_num1 if ans == 0 else self.data_num2
        
        alpha = action[-2:]
        # ------------------------------------------------------------------------------------------
        # transcription = i2u2s2t(action[:-2])
        # ------------------------------------------------------------------------------------------
        pred_num = np.argmax(alpha)
        pred_num = self.data_num1 if pred_num == 0 else self.data_num2
        ans_img = self.get_transformed_image(pred_num)
        transcription = i2u2s2t_img(ans_img)
        # ------------------------------------------------------------------------------------------

        # uses the last 2 action[-2:] to predict the pos of the right ans
        pred_num = np.argmax(alpha)
        print("alpha", alpha, "pred_num", pred_num, "ans", ans, flush=True)
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
        # img_path = img_path.split("/")[-1]
        # right_ans, right_name = judge_ans(transcription, img_path)
        right_ans, right_name = judge_ans(transcription, img_path)
        print(
            self.num_step,
            get_image_info(self.img_list[self.data_num1]),
            get_image_info(self.img_list[self.data_num2]),
            f"ANSWER: {get_image_info(img_path)}",
            flush=True,
            )
        # print(
        #     self.num_step,
        #     get_image_info(self.img_list[self.data_num1].split("/")[-1]),
        #     get_image_info(self.img_list[self.data_num2].split("/")[-1]),
        #     f"ANSWER: {get_image_info(img_path)}",
        #     flush=True,
        #     )
        if right_ans:
            print(self.rewards[0], self.num_step, transcription, flush=True)
            return self.rewards[0]
        elif right_name:
            print(self.rewards[1], self.num_step, transcription, flush=True)
            return self.rewards[1]
        else:
            print(self.rewards[2], self.num_step, transcription, flush=True)
            return self.rewards[2]

        # transcription_ans = img_path.split("/")[5].replace("_", " ").upper()
        # preposition = 'an' if transcription_ans[0] in ['A', 'O', 'E'] else 'a'
        # print(
        #     self.num_step,
        #     self.img_list[self.data_num1].split("/")[5],
        #     self.img_list[self.data_num2].split("/")[5],
        #     f"ANSWER: {transcription_ans}",
        #     flush=True,
        #     )
        # if transcription_ans in transcription:
        #     if f"i want {preposition} {transcription_ans}".upper() == transcription:
        #         print(self.rewards[0], self.num_step, transcription, flush=True)
        #         return self.rewards[0]
        #     else:
        #         print(self.rewards[1], self.num_step, transcription, flush=True)
        #         return self.rewards[1]
        # else:
        #     print(self.rewards[2], self.num_step, transcription, flush=True)
        #     return self.rewards[2]
        
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
    # action_noise = NormalActionNoise(
    #     mean=np.zeros(49*2048+config["u2u"]["d_embed"]+2),
    #     sigma=np.concatenate([np.zeros(49*2048), 0.3*np.ones(config["u2u"]["d_embed"]), np.zeros(2)]),
    #     )
    
    # --------------------------------------------------------------------------------
    # TODO: change the file lists by new divided json files.
    # see /net/papilio/storage2/yhaoyuan/transformer_I2S/data/food_dataset_VC_shuffle.json
    img_data_path = "/net/papilio/storage2/yhaoyuan/transformer_I2S/data/food_dataset_VC_shuffle.json"
    with open(img_data_path, "r") as f:
        img_data = json.load(f)
    img_list_train = [img_data["image_base_path"] + pairdata["image"] for pairdata in img_data["data"]["train"]]
    img_list_eval = [img_data["image_base_path"] + pairdata["image"] for pairdata in img_data["data"]["val"]]
    img_list_test = [img_data["image_base_path"] + pairdata["image"] for pairdata in img_data["data"]["test"]]
    
    if is_debug:
        img_list_train = img_list_train[:10]
        img_list_eval = img_list_eval[:1]
        img_list_test = img_list_test[:1]

    image_hdf5 = f"../../data/RL/256"
    
    # --------------------------------------------------------------------------------
    print("Prepare enviroment")
    # NOTE: changed reward
    env = SpoLacq(
        d_embed=0,
        d_img_features=49*2048,
        # img_list=img_list_train,
        img_hdf5=image_hdf5,
        img_split="TRAIN",
        rewards=[1, 0, 0],
        )
    
    eval_env = SpoLacq(
        d_embed=0,
        d_img_features=49*2048,
        # img_list=img_list_eval,
        img_hdf5=image_hdf5,
        img_split="VAL",
        rewards=[1, 0, 0],
        )
    
    test_env = SpoLacq(
        d_embed=0,
        d_img_features=49*2048,
        # img_list=img_list_test,
        img_hdf5=image_hdf5,
        img_split="TEST",
        rewards=[1, 0, 0],
        )
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
        # action_noise=action_noise,
        replay_buffer_kwargs = dict(handle_timeout_termination=False),
        tensorboard_log='./results/spolacq_tmplog_256_baseline_VC_no_sentence_pos',
        # policy_kwargs=dict(
        #     net_arch=dict(
        #         pi=[[150, 75, 2], [150, 75, config["u2u"]["d_embed"]]],
        #         qf=[150+50+config["u2u"]["d_embed"], (150+50+config["u2u"]["d_embed"])//2, 1],
        #         ),
        #     optimizer_class=torch.optim.AdamW,
        #     optimizer_kwargs=dict(weight_decay=1.0),
        #     )
        policy_kwargs=dict(
            net_arch=dict(
                pi=[[3*features_dim, 3*features_dim, 2], [3*features_dim, 3*features_dim, config["u2u"]["d_embed"]]],
                qf=[4*features_dim, 4*features_dim, 4*features_dim, 1],
                # qf=[4*features_dim+config["u2u"]["d_embed"], (4*features_dim+config["u2u"]["d_embed"])//2, 1],
                ),
            features_extractor_kwargs=dict(features_dim=features_dim),
            normalize_images=False, # We have normalized the img before, no need to normalize again.
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs=dict(weight_decay=1.0),
            )
        )
    # model2.load_param_from_ddpg(CustomDDPG.load("./logs_ddpg_without_embed_icassp/best_model.zip"))
    # model2.fix_param()
    # model2.use_embed()
    # model2.disable_soft_features()
    print("Set up agent complete.")

    print("Start Learning.")
    model2.learn(
        total_timesteps=50000, # 50000
        # total_timesteps=100,
        #'''Removed in version 1.7.0'''
        eval_env=eval_env,
        eval_freq=1000, # 1000
        n_eval_episodes=config["rl"]["n_eval_episodes_ddpg"],
        eval_log_path="./results/logs_ddpg_256_baseline_VC_no_sentence_pos/",
        )
    
    env.save_rewards("./results/spolacq_tmplog_256_baseline_VC_no_sentence_pos/rl_accuracy.npy")