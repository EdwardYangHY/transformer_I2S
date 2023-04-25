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
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import yaml

from scipy.io.wavfile import write

from custom_policy_1_3 import CustomTD3PolicyCNN, CustomDDPG

sys.path.append("../..")
import hifigan
from hifigan.env import AttrDict
from hifigan.models import Generator

sys.path.append("../U2S")
from hparams import create_hparams
from train import load_model
from text import text_to_sequence


sys.path.append("../I2U/models")
# from models import TransformerSentenceLM
from models_modified import TransformerSentenceLM_FixedImg, TransformerSentenceLM_FixedImg_gated
from models_k import TransformerVAEwithCNN

with open('../../config.yml') as yml:
    config = yaml.safe_load(yml)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_debug = True if sys.gettrace() else False

# I2U
print("Preparing I2U model")
word_map_path="../../data/processed/synthesize_speech/WORDMAP_coco_4_cap_per_img_1_min_word_freq.json"
# Load word map (word2ix)
with open(word_map_path) as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
special_words = {"<unk>", "<start>", "<end>", "<pad>"}

# I2U
config["u2u"] = {}
config["u2u"]["d_embed"] = 8
sentence_encoder = TransformerVAEwithCNN(len(word_map), config["u2u"]["d_embed"], max_len = 100)
sentence_encoder.load_state_dict(torch.load("../../saved_model/U2U/transformer_cnn2_synthesize.pt"))
sentence_encoder.eval()
sentence_encoder.to(device)

# config["u2u"] = {}
# config["u2u"]["d_embed"] = 8

# model_path = "../../saved_model/I2U/Komatsu_4_captions_all_224/beam_no_uLM_8_sentence"
# # model_path = "../../saved_model/I2U/Komatsu_4_captions_all_224/beam_gated_uLM_8_sentence"
# # model_path = "../../saved_model/I2U/Komatsu_4_captions_all_224/beam_ungated_uLM_8_sentence"
# word_map_path="/net/papilio/storage2/yhaoyuan/transformer_I2S/data/processed/SpokenCOCO_LibriSpeech/WORDMAP_coco_1_cap_per_img_1_min_word_freq.json"
# # Load word map (word2ix)
# # global word_map, rev_word_map, special_words
# with open(word_map_path) as j:
#     word_map = json.load(j)
# rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
# special_words = {"<unk>", "<start>", "<end>", "<pad>"}

# # i2u_model = load_I2U(model_path, word_map, device)
# config_path = glob(model_path+"/config*.yml")[0]
# model_checkpoint = glob(model_path+"/*BEST*.tar")[0 ]

# with open(config_path, 'r') as yml:
#     model_config = yaml.safe_load(yml)

# model_params = model_config["i2u"]["model_params"]
# model_params['vocab_size'] = len(word_map)
# model_params['refine_encoder_params'] = model_config["i2u"]["refine_encoder_params"]

# params = model_checkpoint.split("/")[-2].split("_")
# if "gated" in params:
#     sentence_encoder = TransformerSentenceLM_FixedImg_gated(**model_params)
# else:
#     sentence_encoder = TransformerSentenceLM_FixedImg(**model_params)
# sentence_encoder.load_state_dict(torch.load(model_checkpoint)["model_state_dict"])

# # sentence_encoder = load_i2u(model_checkpoint, **model_params)
# sentence_encoder.eval()
# sentence_encoder.to(device)



# U2S
print("Preparing U2S model")
hparams = create_hparams()
hparams.sampling_rate = 22050

# tacotron2
checkpoint_path = "../../saved_model/U2S_synthesize/checkpoint_40000"
tacotron2_model = load_model(hparams)
tacotron2_model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
tacotron2_model.cuda().eval()

# HiFi-GAN
# /net/papilio/storage2/yhaoyuan/LAbyLM/dataprep/RL/image2speech_inference.ipynb

checkpoint_file = config["u2s"]['hifigan']
config_file = os.path.join(os.path.split(checkpoint_file)[0], 'config.json')
with open(config_file) as f:
    data = f.read()

# global h
json_config = json.loads(data)
h = AttrDict(json_config)
generator = Generator(h).to(device)
assert os.path.isfile(checkpoint_file)
checkpoint_dict = torch.load(checkpoint_file, map_location=device)
generator.load_state_dict(checkpoint_dict['generator'])
generator.eval()
generator.remove_weight_norm()

# S2T
print("Preparing ASR model")
processor = Wav2Vec2Processor.from_pretrained("../../saved_model/ASR/wav2vec2-synthesize")
asr_model = Wav2Vec2ForCTC.from_pretrained("../../saved_model/ASR/wav2vec2-synthesize").to(device)


def i2u2s2t(action):
    action = torch.from_numpy(action).unsqueeze(0).to(device)
    seq = sentence_encoder.decode(word_map['<start>'], word_map['<end>'], action=action, beam_size=1)
    words = [rev_word_map[ind] for ind in seq if rev_word_map[ind] not in special_words]
    sequence = np.array(text_to_sequence(' '.join(words), ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    try:
        _, mel_outputs_postnet, _, _ = tacotron2_model.inference(sequence)
        with torch.no_grad():
            y_g_hat = generator(mel_outputs_postnet)
            audio = y_g_hat.squeeze()
        audio = audio.cpu().numpy().astype(np.float64)
        # save_audio?
        save_path = "/net/papilio/storage2/yhaoyuan/transformer_I2S/data/RL/RL_generated_wav/"
        # save_wav = audio.astype("int16")
        save_wav = audio
        audio = resampy.resample(audio, 22050, 16000)
        # s2t
        input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.float()
        logits = asr_model(input_values.to(device)).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        save_name = save_path + transcription + ".wav"
        write(save_name, 22050, save_wav)
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
        for img_path in img_list:
            image = Image.open(img_path)
            # image = imread(img_path)
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
        pred_num = np.argmax(alpha)
        print("alpha", alpha, "pred_num", pred_num, "ans", ans, flush=True)
        if ans == pred_num:
            self.rl_rewards.append(1)
        else:
            self.rl_rewards.append(0)

        if transcription[:7] == "I WANT ":
            self.utt_rewards.append(1)
        else:
            self.utt_rewards.append(0)

        img_path = self.img_list[ans_num]
        transcription_ans = img_path.split("/")[4].replace("_", " ").upper()
        preposition = 'an' if transcription_ans[0] in ['A', 'O', 'E'] else 'a'
        print(
            self.num_step,
            self.img_list[self.data_num1].split("/")[4],
            self.img_list[self.data_num2].split("/")[4],
            f"ANSWER: {transcription_ans}",
            flush=True,
            )
        if transcription_ans in transcription:
            if f"i want {preposition} {transcription_ans}".upper() == transcription:
            #if transcription_ans.upper() == transcription:
                print(self.rewards[0], self.num_step, transcription, flush=True)
                
                return self.rewards[0]
            else:
                print(self.rewards[1], self.num_step, transcription, flush=True)
                return self.rewards[1]
        else:
            print(self.rewards[2], self.num_step, transcription, flush=True)
            return self.rewards[2]
    
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


if __name__ == "__main__":
    img_train_list = glob('../../data/image_synthesize/*/train_number*/*.jpg')
    img_eval_list = glob('../../data/image_synthesize/*/test_number[12]/*.jpg')
    img_test_list = glob('../../data/image_synthesize/*/train_number*/*.jpg')

    # if is_debug:
    #     img_train_list = img_train_list[:100]
    #     img_eval_list = img_eval_list[:10]

    print("Set up agent")
    action_noise = NormalActionNoise(
        mean=np.zeros(49*2048+config["u2u"]["d_embed"]+2),
        sigma=np.concatenate([np.zeros(49*2048), 0.2*np.ones(config["u2u"]["d_embed"]), np.zeros(2)]),
        )

    env = SpoLacq(
        d_embed=config["u2u"]["d_embed"],
        d_img_features=49*2048,
        img_list=img_train_list,
        rewards=[1, 0, 0],
        )
    
    eval_env = SpoLacq(
        d_embed=config["u2u"]["d_embed"],
        d_img_features=49*2048,
        img_list=img_eval_list,
        rewards=[1, 0, 0],
        )
    
    features_dim = 2048
    # custom_policy = CustomTD3PolicyCNN(features_extractor_kwargs=feature_extractor_kwargs)

    model2 = CustomDDPG(
        CustomTD3PolicyCNN,
        #custom_policy,
        env,
        learning_rate=config["rl"]["learning_rate"],
        buffer_size=config["rl"]["buffer_size"],
        learning_starts=config["rl"]["learning_starts"],
        batch_size=config["rl"]["batch_size"],
        train_freq=4,
        action_noise=action_noise,
        replay_buffer_kwargs = dict(handle_timeout_termination=False),
        tensorboard_log=f'./spolacq_tmplog_synthesize_test_{features_dim}/',
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
    print(f"feature dim: {features_dim}")
    model2.learn(
        total_timesteps=50000,
        # total_timesteps=100,
        #'''Removed in version 1.7.0'''
        eval_env=eval_env,
        eval_freq=1000,
        n_eval_episodes=config["rl"]["n_eval_episodes_ddpg"],
        eval_log_path=f"./logs_reproduce_test_{features_dim}/",
        )
    
    env.save_rewards(f"./spolacq_tmplog_synthesize_test_{features_dim}/rl_accuracy.npy")



    # test_env = SpoLacq(
    #     d_embed=config["u2u"]["d_embed"],
    #     d_img_features=49*2048,
    #     img_list=glob('../../data/I2U/image_synthesize/*/test_number3/*.jpg'),
    #     rewards=[1, 0, 0],
    #     )

    # model2 = CustomDDPG(
    #     CustomTD3PolicyCNN,
    #     env,
    #     learning_rate=config["rl"]["learning_rate"],
    #     buffer_size=config["rl"]["buffer_size"],
    #     learning_starts=config["rl"]["learning_starts"],
    #     batch_size=config["rl"]["batch_size"],
    #     action_noise=action_noise,
    #     replay_buffer_kwargs = dict(handle_timeout_termination=False),
    #     tensorboard_log='./spolacq_tmplog/',
    #     policy_kwargs=dict(
    #         net_arch=dict(
    #             # pi=[[768*3, 768*3, 2], [768*3, 768*3, config["u2u"]["d_embed"]]],
    #             # qf=[768*3+768+config["u2u"]["d_embed"], 768*3+768+config["u2u"]["d_embed"], 768*3+768+config["u2u"]["d_embed"], 1],
    #             pi=[[2048*3, 2048*3, 2], [2048*3, 2048*3, config["u2u"]["d_embed"]]],
    #             qf=[2048*4+config["u2u"]["d_embed"], 2048*4+config["u2u"]["d_embed"], 2048*4+config["u2u"]["d_embed"], 1],
    #             ),
    #         optimizer_class=torch.optim.AdamW,
    #         optimizer_kwargs=dict(weight_decay=1.0),
    #         )
    #     )
    # model2.use_embed()

    # model2.learn(
    #     total_timesteps=500000,
    #     eval_env=eval_env,
    #     eval_freq=200,
    #     n_eval_episodes=config["rl"]["n_eval_episodes_ddpg"],
    #     eval_log_path="./logs_ddpg_thesis_wide_cnn_single_word_action_noise_0_2_500000/",
    #     )
    
    # eval_env.save_rewards("./logs_ddpg_thesis_wide_cnn_single_word_action_noise_0_2_500000/rl_accuracy.npy")