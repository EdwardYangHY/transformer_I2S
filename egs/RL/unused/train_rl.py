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

from my_custom_policy2 import CustomTD3Policy, CustomDDPG

sys.path.append('../../dataprep/RL')
from i2u2s2t5 import i2u2s2t

with open('../../config.yml', 'r') as yml:
    config = yaml.safe_load(yml)

#画像fileの一覧を取得
train_lists = glob('../../data/I2U/image/*/train_number*/*.jpg')
val_lists = glob('../../data/I2U/image/*/test_number[12]/*.jpg')
test_lists = glob('../../data/I2U/image/*/test_number[3]*/*.jpg')

#rgb取得
def get_rgb(img_path):
    img = Image.open(img_path)
    img = np.asarray(img)
    r,g,b = 0,0,0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r+=img[i][j][0]
            g+=img[i][j][1]
            b+=img[i][j][2]
    pixels = img.shape[0] * img.shape[1]
    return [r//pixels,g//pixels,b//pixels]

#正解判定関数
def get_ans(file_num,file_num2,inner_state,filelists):
    rgb_1 = get_rgb(filelists[file_num])
    rgb_2 = get_rgb(filelists[file_num2])
    rgb_ans = inner_state
    distance_1 = sum(list(map(lambda x,y:(x-y)^2,rgb_1,rgb_ans)))
    distance_2 = sum(list(map(lambda x,y:(x-y)^2,rgb_2,rgb_ans)))
    return 0 if distance_1 < distance_2 else 1


#reward関数
def get_reward(file_num,file_num2,inner_state,predict,filelists):
    return 1 if predict == get_ans(file_num,file_num2,inner_state,filelists) else 0

NUM_STEP = 0
def get_recog_reward(file_num,file_num2,inner_state,action,filelists,mode,use_embed):
    global NUM_STEP
    NUM_STEP += 1
    
    ans = get_ans(file_num,file_num2,inner_state,filelists)
    
    ans_num = file_num if ans == 0 else file_num2
    img_path = filelists[ans_num]

    transcription = i2u2s2t(action, use_embed=use_embed)
    
    ans = img_path.split("/")[5].split("_")[-1].upper()
    ans2 = img_path.split("/")[5].split("_")[0].upper()

    if ans in transcription.split() and ans2 in transcription.split():
        rl_rewards[NUM_STEP-1] = 1
        
        if "I WANT " in transcription:
            rewards[NUM_STEP-1] = 1
            utt_rewards[NUM_STEP-1] = 1
            
            print(1, NUM_STEP, transcription, flush=True)
            return 1
        else:
            print(1, NUM_STEP, transcription, flush=True)
            return 1
    else:
        if "I WANT " in transcription:
            utt_rewards[NUM_STEP-1] = 1

        print(0, NUM_STEP, transcription, flush=True)
        return 0
#reward関数　ユニット版

#画像データ取得
def get_np_image(img_path):
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image = Image.open(img_path)
    image = np.asarray(image)
    image = image.transpose(2, 0, 1)
    image = torch.FloatTensor(image / 255.)
    image = transform(image)
    image = image.numpy()
    return image


class SpoLacq(gym.Env):
    def __init__(self, use_embed: bool):
        super().__init__()
        d_embed = config["u2u"]["d_embed"]
        self.use_embed = use_embed
        if use_embed:
            self.action_space = gym.spaces.Box(low=-np.inf,high=np.inf,shape=(7*7*2048+d_embed,))
        else:
            self.action_space = gym.spaces.Box(low=-np.inf,high=np.inf,shape=(7*7*2048,))
        self.observation_space = gym.spaces.Dict(
            dict(
                state=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                leftimage=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3, 224, 224), dtype=np.float64),
                rightimage=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3, 224, 224), dtype=np.float64),
            )
        )
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.reset()

    # 環境のリセット
    def reset(self):
        # 初期位置の指定
        self.data_num1 = random.randint(0, len(train_lists)-1)
        self.data_num2 = random.randint(0, len(train_lists)-1)
        self.inner_state = np.array([random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)])
        print(train_lists[self.data_num1].split("/")[5], train_lists[self.data_num2].split("/")[5], flush=True)
        data1 = get_np_image(train_lists[self.data_num1])
        data2 = get_np_image(train_lists[self.data_num2])
        input_rl = dict(
            state=(self.inner_state/255.-self.mean) / self.std,
            leftimage=data1,
            rightimage=data2,
        )
        return input_rl

    # 環境の１ステップ実行
    def step(self, action):
        reward = get_recog_reward(self.data_num1,self.data_num2,self.inner_state,action,train_lists,"train",self.use_embed)
        self.data_num1 = random.randint(0, len(train_lists)-1)
        self.data_num2 = random.randint(0, len(train_lists)-1)
        data1 = get_np_image(train_lists[self.data_num1])
        data2 = get_np_image(train_lists[self.data_num2])
        self.inner_state = np.array([random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)])
        input_rl = dict(
            state=(self.inner_state/255.-self.mean) / self.std,
            leftimage=data1,
            rightimage=data2,
        )
        return input_rl, reward, True, {}


#評価for文
#精度測定
#グラフ作成
def test(filelists,n_calls=None):
    acc = 0
    acc_rl = 0
    test_num = 0
    #左の画像は全て、右の画像はランダム
    for data_num1 in range(len(filelists)):
        inner_state = np.array([random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)])
        data1 = get_np_image(filelists[data_num1])
        data_num2 = random.randint(0, len(filelists)-1)
        #同じ画像が選ばれないように
        while data_num1 == data_num2:
            data_num2 = random.randint(0, len(filelists)-1)
        data2 = get_np_image(filelists[data_num2])
        input_rl = dict(
                state=inner_state,
                leftimage=data1,
                rightimage=data2,
            )
        test_num += 1
        if n_calls != None:
            acc += get_recog_reward(data_num1,data_num2,inner_state,model.predict(input_rl)[0],val_lists,"val")
            acc_rl += get_reward(data_num1,data_num2,inner_state,model.predict(input_rl)[0],val_lists)
        else:
            acc += get_recog_reward(data_num1,data_num2,inner_state,model.predict(input_rl)[0],test_lists,"test")
            acc_rl += get_reward(data_num1,data_num2,inner_state,model.predict(input_rl)[0],test_lists)
    accurate = acc/test_num
    accurate_rl = acc_rl/test_num
    #結果ファイルに保存
    if n_calls != None:
        f = open(config["rl"]["result"], 'a')
        f.write(f'{accurate*100}, {n_calls}\n')
        f.close()
        f = open(config["rl"]["result_rl"], 'a')
        f.write(f'{accurate_rl*100}, {n_calls}\n')
        f.close()
    return f"acc: {acc}/{test_num}\nacc_rl: {acc_rl}/{test_num}"


# モデルの生成
env = SpoLacq(use_embed=False)
model = CustomDDPG(
    CustomTD3Policy,
    env,
    buffer_size=1000,
    replay_buffer_kwargs = dict(handle_timeout_termination=False),
    tensorboard_log='./spolacq_tmplog/',
    )
print("train start")

# モデルの学習
total_timesteps = 100000
rewards = np.zeros(total_timesteps, dtype=np.uint8)
utt_rewards = np.zeros(total_timesteps, dtype=np.uint8)
rl_rewards = np.zeros(total_timesteps, dtype=np.uint8)
model.learn(total_timesteps=total_timesteps)
data_name = datetime.datetime.now().strftime('%y%m%d%H%M')
model.save(config["rl"]["model"])
np.save(f"../../model/RL/rewards_{data_name}.npy", rewards)
np.save(f"../../model/RL/utt_rewards_{data_name}.npy", utt_rewards)
np.save(f"../../model/RL/rl_rewards_{data_name}.npy", rl_rewards)

# モデルのテスト
# print(f"test_acc: {test(test_lists)}")
# state = env.reset()
# total_reward = 0
# for i in range(30):
#   action, _ = model.predict(state)
#   state, reward, done, info = env.step(action)
#   print(action,reward)
# total_reward += reward


def plot(rewards_path: str):
    rewards = np.load(rewards_path)
    episodes = len(rewards)
    duration = 1000
    moving_average = np.zeros(episodes//duration)
    for i in range(episodes//duration):
        moving_average[i] = np.mean(rewards[duration*i:duration*(i+1)])
    
    plt.plot(list(range(1, 1 + episodes//duration)), moving_average)
    plt.show()