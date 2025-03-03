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
num_episodes = 100  # 每个episode的迭代次数
alpha = 0.0003  # learning rate学习率
fc1_dim = 256  # 神经网络第一层的神经元数量
fc2_dim = 256  # 神经网络第二层的神经元数量
gamma = 0.92  # discount factor
tau = 0.05  # soft update parameter
epsilon = 1.0  # epsilon初始值
eps_end = 0.05  # 学习过程中 epsilon 值将逐渐减小到的最终值
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
# GridWorld-v10

# env = gym.make('TestWorld-v3')
env = gym.make(env_name)
agent = DDQN(alpha=alpha, state_dim=4, action_dim=8,
             fc1_dim=fc1_dim, fc2_dim=fc2_dim, ckpt_dir=args.ckpt_dir, gamma=gamma, tau=tau, epsilon=epsilon,
             eps_end=eps_end, eps_dec=eps_dec, max_size=max_size, batch_size=batch_size)
agent.load_models(eval_path, target_path)
agent.q_eval.eval()
agent.q_target.eval()


def dic_to_list(dic):

    list_ = [list(dic['agent_x']), list(dic['agent_y']),
             list(dic['T_D']), list(dic['O_D'])]
    array = np.array(list_)
    # 执行转置操作以交换行和列
    transposed_array = array.T
    return transposed_array


def main():
    avg_reward_list, total_reward_list = [], []
    for episode in range(args.max_episodes):
        done = False
        state_dic = env.reset(isTest=True)
        observation = dic_to_list(state_dic)
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
            observation_ = dic_to_list(state_dic)
            # print(reward)
            reward = torch.tensor(
                [reward], dtype=torch.long, device=device)
            reward_this_epi = reward_this_epi + reward.item()
            if done or truncated:
                save_path_as_image(map_, episode)
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
        # print(total_reward_list, avg_reward, avg_reward_list)



def save_path_as_image(map_, episode):
    # 首先，我们调用render方法获取画布的numpy数组表示
    # canvas_array = self.render(mode="rgb_array")
    # 然后，我们使用matplotlib或PIL等库将numpy数组保存为图片文件
    # 这里我们使用PIL作为示例
    # from PIL import Image

    # 将numpy数组转换为PIL图像对象
    image = Image.fromarray(map_)
    # 保存图像到文件，你可以指定保存的路径和文件名
    image.save(f"{PATH}path_to_target.png__{episode}.jpg")
    print("Path saved as image!")


if __name__ == '__main__':
    print("-------------Training :-------------")
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('CUDA是否可用:', torch.cuda.is_available())
    # 列出可用的CUDA设备
    print('CUDA设备数量:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'设备{i}名称:', torch.cuda.get_device_name(i))
    main()
