from glob import glob

import yaml

from custom_policy import CustomTD3Policy, CustomDDPG
# from train_ddpg2 import SpoLacq
from train_ddpg import SpoLacq

with open('../../config.yml') as yml:
    config = yaml.safe_load(yml)

test_env = SpoLacq(
    # d_embed=config["u2u"]["d_embed"],
    d_embed=0,
    d_img_features=768,
    img_list=glob('../../data/I2U/image/*/test_number3/*.jpg'),
    rewards=[1, 0, 0],
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


model = CustomDDPG.load("./logs_ddpg_without_embed_icassp/best_model.zip")
# model.use_embed()
test(test_env, model)