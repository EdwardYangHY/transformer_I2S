import gym
import numpy as np
from PIL import Image
from imageio import imread
import resampy
import torch
import h5py
from tqdm import tqdm
from torchvision import transforms

name_dict={
    'apple': ['apple','apples'],
    'banana': ['banana','bananas'],
    'carrot': ['carrot','carrots'],
    'grape': ['grape','grapes'],
    'cucumber': ['cucumber','cucumbers'],
    'egg': ['egg','eggs'],
    'eggplant': ['eggplant','eggplants'],
    'greenpepper': ['pepper','peppers','green pepper','green peppers'],
    'pea': ['pea','peas','green pea','green peas'],
    'kiwi': ['kiwi','kiwi fruit','kiwi fruits'],
    'lemon': ['lemon','lemons'],
    'onion': ['onion','onions'],
    'orange': ['orange','oranges'],
    'potatoes': ['potato','potatoes'],
    'bread': ['bread', 'sliced bread'],
    'avocado': ['avocado','avocados'],
    'strawberry': ['strawberry','strawberries'],
    'sweetpotato': ['sweet','sweet potato','sweet potatoes'],
    'tomato': ['tomato','tomatoes'],
    'turnip': ['radish','radishes','white radish','white radishes']
    #'orange02': '/orange02'
}
color_dict = {
    'wh': 'white',
    'br': 'brown',
    'bl': 'blue'
}
number_dict = {
    1: 'one',
    2: 'two',
    3: 'three'
}

def get_image_info(image_path):
    # info=image_path.split("/")[8]
    info = image_path.split("/")[-1]
    name=info.split("_")[0]
    if name=="orange":
        color=info.split("_")[1]
        number=info.split("_")[2]
    else:
        color=info.split("_")[1][0:-1]
        number=info.split("_")[1][-1]
    return name, color, number

def judge_ans(transcription, img_path):
    ans = transcription.split(" ")
    # print(ans)
    right_name = False
    right_color = False
    right_number = False
    right_ans = False

    name, color, number = get_image_info(img_path)

    # 分开两个词 怎么办
    for an in ans:
        if an in name_dict[name]:
            # print("Right name.")
            right_name = True
        if an == color_dict[color]:
            # print("Right color.")
            right_color = True
        if an in number_dict[int(number)]:
            # print("Right number.")
            right_number = True
    if right_name and right_color and right_number:
        right_ans = True
    return right_ans, right_name

# --------------------------------------------------------------------------------

class SpoLacq(gym.Env):
    """
        Original Spoken Langauge Acquisition env.
        The agent takes 2 image features as input (shaped by ResNet, (224/32)*(224/32)*2048)
        And it has a randomly generated internal "preferance" (RGB value of 255).
        The true answer is the selected image's RGB with closer distance.
        According to given image, the agent has to pronouce:
        "I want ___".
        Only answering food name will not be rewarded.

        param d_embed: the size of the sentence embedding. Usually 0, 8, 16
        param d_img_features: The size of the image feature dim. Default 2048.
        param img_list: List object of image paths. (if hdf5 available, it can be only img names).
        param img_hdf5: Already processed imgs (resize and permuted to [3, 224, 224]) in hdf5 file.
        param i2u2s2t: Method. I2U model's decode function, takes action as input, output 
                       transcriptions.
        param rewards: List object of rewards.                       
    """
    def __init__(
        self,
        d_embed: int,
        d_img_features: int,
        img_list: list,
        img_hdf5 = None,
        i2u2s2t = None,
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
        self.use_hdf5 = False
        if img_hdf5 is not None:
            self.use_hdf5 = True
            self.make_food_storage_hdf5(img_hdf5)
        else:
            self.make_food_storage(img_list)
        
        self.rewards = rewards
        self.utt_rewards = list()
        self.rl_rewards = list()
        self.i2u2s2t = i2u2s2t
        
        self.num_step = 0
        self.reset()
    
    def make_food_storage_hdf5(self, img_hdf5):
        self.food_storage = list()
        h = h5py.File(img_hdf5, "r")
        images = h['images']
        for img in images:
            self.food_storage.append(img)
    
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
        if not self.use_hdf5:
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
        if not self.use_hdf5:
            rgb_1 = np.mean(self.food_storage[self.data_num1], axis=(0,1))
            rgb_2 = np.mean(self.food_storage[self.data_num2], axis=(0,1))
        else:
            rgb_1 = np.mean(self.food_storage[self.data_num1], axis=(1,2))
            rgb_2 = np.mean(self.food_storage[self.data_num2], axis=(1,2))
        distance_1 = np.linalg.norm(rgb_1-self.internal_state)
        distance_2 = np.linalg.norm(rgb_2-self.internal_state)
        return 0 if distance_1 < distance_2 else 1

    def reward(self, action):
        """
            Special designed Rewards funtion.
            action is selected, flatten image features (plus sentence embedding) and 2-dim Alpha.
            Alpha is like a position selector: 
                Inside the action selection, the alpha is softmaxed, and as weights to make soft
                image features combining both in training.
        """
        self.num_step += 1
        ans = self.get_ans()
        ans_num = self.data_num1 if ans == 0 else self.data_num2
        transcription = self.i2u2s2t(action[:-2])

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



class SpoLacq_new(SpoLacq):
    """
        Only change the reward function for new task setting: colors
    """
    def __init__(self, d_embed: int, d_img_features: int, img_list: list, img_hdf5=None, i2u2s2t=None, rewards: list = [1, 0, 0]):
        super().__init__(d_embed, d_img_features, img_list, img_hdf5, i2u2s2t, rewards)

    def reward(self, action):
        self.num_step += 1
        ans = self.get_ans()
        ans_num = self.data_num1 if ans == 0 else self.data_num2
        transcription = self.i2u2s2t(action[:-2])
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