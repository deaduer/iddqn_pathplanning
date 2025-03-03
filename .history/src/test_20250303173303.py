# 测试模型tqn_9
# 加载保存的模型

import gymnasium as gym
import numpy as np
import argparse

from torch.utils.tensorboard import SummaryWriter
import torch
from PIL import Image

from DDQN import DDQN

# [213, 266, 277, 282]
# 是否加入扰动
disturbance = False
PATH = ''  
env_name = ''  
eval_name = ''  
target_name = ''   
ith_sample = '.pth'  
eval_path = PATH + eval_name + ith_sample

target_path = PATH + target_name + ith_sample



iteration = 10 
num_episodes = 100  
alpha = 0.0003  
fc1_dim = 256  
fc2_dim = 256  
gamma = 0.92  
tau = 0.05  
epsilon = 1.0  
eps_end = 0.05  
eps_dec = 5e-4  
max_size = 10000 
batch_size = 64  
device = 'cuda'


parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=num_episodes)
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/DDQN/')
parser.add_argument('--reward_path', type=str,
                    default='./output_images/avg_reward.png')
parser.add_argument('--epsilon_path', type=str,
                    default='./output_images/epsilon.png')
args = parser.parse_args()



env = gym.make(env_name)
agent = DDQN(alpha=alpha, state_dim=4, action_dim=8,
             fc1_dim=fc1_dim, fc2_dim=fc2_dim, ckpt_dir=args.ckpt_dir, gamma=gamma, tau=tau, epsilon=epsilon,
             eps_end=eps_end, eps_dec=eps_dec, max_size=max_size, batch_size=batch_size)
agent.load_models(eval_path, target_path)
agent.q_eval.eval()
agent.q_target.eval()





def main():
    avg_reward_list, total_reward_list = [], []
    for episode in range(args.max_episodes):
        done = False
        state_dic = env.reset(isTest=True)
        observation = state_dic
        reward_this_epi = 0
        total_reward = 0
        num_action = 0
        while not done:
            map_ = env.render()
            num_action += 1
            if disturbance:  # 是否加入扰动
                if num_action % 5 == 0:  # 每5步加入扰动
                    action = 0
                else:
                    action = agent.choose_action(
                        observation, episode, isTrain=False)
            else:
                action = agent.choose_action(
                    observation, episode, isTrain=False)
            state_dic, reward, done, truncated, info = env.step(action)
            total_reward += reward
            observation_ = state_dic
        
            reward = torch.tensor(
                [reward], dtype=torch.long, device=device)
            reward_this_epi = reward_this_epi + reward.item()
            if done or truncated:
                
                if reward == 1000:
                    print('goal')
                else:
                    print('fail')

                break
            else:

                observation = observation_
        total_reward_list.append(total_reward)
        avg_reward = np.mean(total_reward_list[-100:])
        avg_reward_list.append(avg_reward)







if __name__ == '__main__':
    print("-------------Training :-------------")
    main()
