'''
Description: 自适应搜索策略训练
Version: 2.0
Author: zwz
Date: 2024-10-28 16:42:53
LastEditors: zwz
LastEditTime: 2024-11-09 15:27:09
'''

# ddqn zwz 2024.10.14

# TODO势场大小加入reward里
# TODO N-step 用正常随机抽样要修改take_action 以及q值计算公式
import datetime
import argparse
import gymnasium as gym
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from DDQN import DDQN
from utils import plot_learning_curve, create_directory
from colorama import Fore, Back, Style, init
init(autoreset=True)

current_month_day = datetime.datetime.now().strftime("%m_%d")
PATH = f''
env_name = ''
print(Fore.BLUE + f'{env_name}')

save_model = True  
epoch = 1  
num_episodes = 50000  
alpha = 0.003  
fc1_dim = 256  
fc2_dim = 256  
gamma = 0.9  

tau = 0.05  # soft update parameter

epsilon = 1.0  # epsilon初始值
eps_end = 0.05  # 学习过程中 epsilon 值将逐渐减小到的最终值
# eps_dec = 5e-4  # 控制了 epsilon 值在学习过程中减小的速度
eps_dec = 500  # 控制了 epsilon 值在学习过程中减小的速度 # TODO引入论文里的 epsilon 衰减策略
max_size = 10000  # replay buffer的大小
batch_size = 64  # batch size取样大小
n_multi_step = 3  # N-step





env = gym.make(env_name)
env.reset()
state_dic, reward, done, truncated, info = env.step(
    env.action_space.sample())

n_actions = env.action_space.n
state_dim = len(state_dic)  # 状态空间维度
env.close()


def dic_to_list(dic):

    list_ = [list(dic['agent_x']), list(dic['agent_y']),
             list(dic['T_D']), list(dic['O_D'])]
    array = np.array(list_)
    # 执行转置操作以交换行和列
    transposed_array = array.T
    return transposed_array


def main():
    agent = DDQN(alpha=alpha, state_dim=state_dim, action_dim=n_actions,
                 fc1_dim=fc1_dim, fc2_dim=fc2_dim, ckpt_dir=args.ckpt_dir, gamma=gamma, tau=tau, epsilon=epsilon,
                 eps_end=eps_end, eps_dec=eps_dec, max_size=max_size, batch_size=batch_size, n_multi_step=n_multi_step, num_episodes=args.max_episodes)
    create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])


    for i in range(epoch):
        
        with tqdm(total=int(num_episodes), desc='Epoch %d' % i) as pbar:
            for episode in range(args.max_episodes):
                total_reward = 0
                done = False
                state_dic = env.reset()
                observation = dic_to_list(state_dic)
                while not done:

                    action = agent.choose_action(
                        observation, episode, isTrain=True)
                    state_dic, reward, done, truncated, info = env.step(action)
                    observation_ = dic_to_list(state_dic)
                    agent.remember(
                        observation[-1], action, reward, observation_[-1], done)
                    agent.learn()
                    total_reward += reward
                    observation = observation_

                



                pbar.update(1)

            



eval_name = '\\Q_eval\\DDQN_q_eval_'  # 请替换为您的测试名称
target_name = '\\Q_target\\DDQN_Q_target_'  # 请替换为您的目标名称

filename = "test_record.txt"

with open(filename, "w") as file:
    file.write(f"{PATH}test_record_{current_month_day}_{env_name}\n")




if __name__ == '__main__':
    print(Back.YELLOW + Fore.BLACK +
          "---------------------------------------Training :---------------------------------------")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(Fore.RED + 'CUDA是否可用:',  torch.cuda.is_available())

    print(Fore.RED + 'CUDA设备数量:',  torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(Fore.RED + f'设备{i}名称:',  torch.cuda.get_device_name(i))
    main()
