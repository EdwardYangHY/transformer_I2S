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

sys.path.append("../I2U")
from utils_synthesize import *
from judge_asr import judge_ans, get_image_info

# config path需要更改
with open('../../config.yml') as yml:
    config = yaml.safe_load(yml)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_debug = True if sys.gettrace() else False

# --------------------------------------------------------------------------------
print("Load u2s2t model")
tts_model_path = "/net/papilio/storage2/yhaoyuan/transformer_I2S/gslm_models/u2S/HuBERT_KM100_tts_checkpoint_best.pt"
max_decoder_steps = 500
code_dict_path = "/net/papilio/storage2/yhaoyuan/transformer_I2S/gslm_models/u2S/HuBERT_KM100_code_dict"

hifigan_checkpoint_path = config["u2s"]['hifigan']
asr_checkpoint_path = config["asr"]["model_path"]

# tacotron_model = load_tacotron(tacotron_checkpoint_path, tacotron_max_decoder_step)
tacotron_model, tts_datasets = load_tacotron2_hubert(model_path=tts_model_path, code_dict_path=code_dict_path, max_decoder_steps=max_decoder_steps)
hifigan_model = load_hifigan(hifigan_checkpoint_path, device)
asr_model, asr_processor = load_asr(asr_checkpoint_path, device)

# --------------------------------------------------------------------------------

i2u_checkpoint = "../../saved_model/I2U/origin_5_captions_256_hubert/hubert_baseline/bleu-4_BEST_checkpoint_coco_5_cap_per_img_1_min_word_freq_gpu.pth.tar"
i2u_config_path = "../../saved_model/I2U/origin_5_captions_256_hubert/hubert_baseline/config_sentence.yml"
word_map_path = "/net/papilio/storage2/yhaoyuan/transformer_I2S/data/processed/origin_5_captions_256_hubert/WORDMAP_coco_5_cap_per_img_1_min_word_freq.json"
with open(word_map_path) as f:
    word_map = json.load(f)
rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
special_words = {"<unk>", "<start>", "<end>", "<pad>"}

i2u_model = load_i2u(i2u_checkpoint, i2u_config_path, len(word_map))
i2u_model.eval()
i2u_model.to(device)
# --------------------------------------------------------------------------------

def i2u2s2t(action):
    seqs = i2u_model.decode(action=action, start_unit=word_map["<start>"], end_unit=word_map["<end>"], max_len=150, beam_size=10)
        
    words = seq2words(seq=seqs, rev_word_map=rev_word_map, special_words=special_words)
    # audio = u2s(
    #     words=words,
    #     tacotron2_model=tacotron_model
    #     hifigan_model=hifigan_model,
    #     device=device
    #     )
    audio = u2s_hubert(
        words=words,
        tacotron2_model=tacotron_model,
        tts_dataset=tts_datasets,
        hifigan_model=hifigan_model,
        device=device
        )
    
    trans = s2t(audio=audio, asr_model=asr_model, asr_processor=asr_processor, device=device)
    return trans
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
                leftimage=gym.spaces.Box(low=-100, high=100, shape=(3, 224, 224), dtype=np.float64),
                rightimage=gym.spaces.Box(low=-100, high=100, shape=(3, 224, 224), dtype=np.float64),
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
        right_ans, right_name = judge_ans(transcription, img_path)
        print(
            self.num_step,
            get_image_info(self.img_list[self.data_num1]),
            get_image_info(self.img_list[self.data_num2]),
            f"ANSWER: {get_image_info(img_path)}",
            flush=True,
            )
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
    action_noise = NormalActionNoise(
        mean=np.zeros(49*2048+config["u2u"]["d_embed"]+2),
        sigma=np.concatenate([np.zeros(49*2048), 0.3*np.ones(config["u2u"]["d_embed"]), np.zeros(2)]),
        )
    
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
    
    # foods_incremental = [
    #     'lemon',
    #     'onion',
    #     'orange',
    #     'potato',
    #     'sliced_bread',
    #     'small_cabbage',
    #     'strawberry',
    #     'sweet_potato',
    #     'tomato',
    #     'white_radish',
    # ]

    # img_list_train = glob('../../data/I2U/image/*/train_number*/*.jpg')
    # img_list_eval = glob('../../data/I2U/image/*/test_number[12]/*.jpg')
    # img_list_test = glob('../../data/I2U/image/*/test_number3/*.jpg')

    # if config["rl"]["incremental"]:
    #     img_list_train = [path for path in img_list_train if path.split("/")[5] not in foods_incremental]
    #     img_list_eval = [path for path in img_list_eval if path.split("/")[5] not in foods_incremental]
    #     img_list_test = [path for path in img_list_test if path.split("/")[5] not in foods_incremental]

    # --------------------------------------------------------------------------------
    print("Prepare enviroment")
    # NOTE: changed reward
    env = SpoLacq(
        d_embed=config["u2u"]["d_embed"],
        d_img_features=49*2048,
        img_list=img_list_train,
        rewards=[1, 0.1, 0],
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
        tensorboard_log='./spolacq_tmplog_new_fixed_image_8_sentence/',
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
        eval_log_path="./logs_ddpg_new_fixed_image_8_sentence/",
        )
    
    env.save_rewards("./spolacq_tmplog_new_fixed_image_8_sentence/rl_accuracy.npy")