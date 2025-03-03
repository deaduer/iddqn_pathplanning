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

save_model = True  # 是否保存模型
epoch = 1  # 训练的总轮数
num_episodes = 50000  # 每个episode的迭代次数
alpha = 0.003  # learning rate学习率0.0003->0.003
fc1_dim = 256  # 神经网络第一层的神经元数量
fc2_dim = 256  # 神经网络第二层的神经元数量
gamma = 0.9  
'''
折扣因子反映了未来奖励的重要性。
一个接近1的 gamma 值意味着算法非常重视长期的未来奖励，
而一个接近0的值则意味着算法更关注即时的奖励。
'''
tau = 0.05  # soft update parameter
'''
通常用于目标网络Target Network的软更新过程中。
在深度Q学习或DDPG等算法中,为了稳定学习过程,通常会维护一个与主网络Main Network结构相同但参数更新较慢的目标网络。
'''
epsilon = 1.0  # epsilon初始值
eps_end = 0.05  # 学习过程中 epsilon 值将逐渐减小到的最终值
# eps_dec = 5e-4  # 控制了 epsilon 值在学习过程中减小的速度
eps_dec = 500  # 控制了 epsilon 值在学习过程中减小的速度 # TODO引入论文里的 epsilon 衰减策略
max_size = 10000  # replay buffer的大小
batch_size = 64  # batch size取样大小
n_multi_step = 3  # N-step
'''
reward很稀疏的时候，one-step TD只有当sample到的一个transition具有reward信息才能学习，而N-step bootstrap可以sample到N个transition，
其中只要有具有reward信息的transition，就可以学习。因此，N-step bootstrap会有更好的学习效率。
'''


parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=num_episodes)
parser.add_argument('--ckpt_dir', type=str, default=PATH)
parser.add_argument('--reward_path', type=str,
                    default='./output_images/avg_reward.png')
parser.add_argument('--epsilon_path', type=str,
                    default='./output_images/epsilon.png')
args = parser.parse_args()


env = gym.make(env_name)
env.reset()
state_dic, reward, done, truncated, info = env.step(
    env.action_space.sample())
# print(state_dic)
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
    total_rewards, avg_rewards, eps_history = [], [], []
    model_num = 0  # 模型编号
    for i in range(epoch):
        writer = SummaryWriter(f'DDQN{PATH}_{env_name}')
        num_goal = 0
        success_rate = 0
        with tqdm(total=int(num_episodes), desc='Epoch %d' % i) as pbar:
            for episode in range(args.max_episodes):
                total_reward = 0
                done = False
                state_dic = env.reset()
                observation = dic_to_list(state_dic)
                while not done:
                    # env.render()
                    action = agent.choose_action(
                        observation, episode, isTrain=True)
                    state_dic, reward, done, truncated, info = env.step(action)
                    observation_ = dic_to_list(state_dic)
                    agent.remember(
                        observation[-1], action, reward, observation_[-1], done)
                    loss = agent.learn()
                    total_reward += reward
                    observation = observation_
                    if done or truncated:
                        if reward == 1000:
                            num_goal += 1

                success_rate = num_goal / (episode + 1)
                total_rewards.append(total_reward)
                avg_reward = np.mean(total_rewards)
                avg_rewards.append(avg_reward)
                eps_history.append(agent.epsilon)
                if (episode + 1) % 10 == 0:
                    pbar.set_postfix({'avg_reawrd': '%f' % (avg_reward), 'total_reward': '%f' % (total_reward), 'success_rate': '%f' % (success_rate), 'goal':
                                      '%d' % (num_goal)
                                      })
                if save_model and ((episode+1) % 100 == 0):
                    agent.save_models(model_num)
                    test(model_num)
                    model_num += 1

                pbar.update(1)
                writer.add_scalar(
                    f'average_reward', avg_reward, episode)
                writer.add_scalar(
                    f'success_rate', success_rate, episode)
                writer.add_scalar(
                    f'total_reward', total_reward, episode)
            reward_std = np.std(total_rewards)
            print(Fore.RED + 'reward_std:', reward_std)
            # if loss is None:
            #     loss = 0

            # avg_reward_iteration=np.mean(total_rewards)
            # writer.add_scalar(
            #     f'average_reward_iteration', avg_reward_iteration, epoch)
            # if (episode + 1) % 50 == 0:
            #     agent.save_models(episode + 1)
            # episodes = [i for i in range(args.max_episodes)]
            # plot_learning_curve(episodes, avg_rewards, 'Reward',
            #                     'reward', args.reward_path)
            # plot_learning_curve(episodes, eps_history, 'Epsilon',
            #                     'epsilon', args.epsilon_path)


eval_name = '\\Q_eval\\DDQN_q_eval_'  # 请替换为您的测试名称
target_name = '\\Q_target\\DDQN_Q_target_'  # 请替换为您的目标名称

filename = "test_record.txt"

with open(filename, "w") as file:
    file.write(f"{PATH}test_record_{current_month_day}_{env_name}\n")


def test(episode):
    ith_sample = f'{episode}.pth'  # 假设您的样本后缀是这样，请根据实际情况替换
    eval_path = PATH + eval_name + ith_sample
    target_path = PATH + target_name + ith_sample
    env = gym.make(env_name)
    agent_test = DDQN(alpha=alpha, state_dim=4, action_dim=8,
                      fc1_dim=fc1_dim, fc2_dim=fc2_dim, ckpt_dir=args.ckpt_dir, gamma=gamma, tau=tau, epsilon=epsilon,
                      eps_end=eps_end, eps_dec=eps_dec, max_size=max_size, batch_size=batch_size)
    agent_test.load_models(eval_path, target_path)
    agent_test.q_eval.eval()
    agent_test.q_target.eval()
    test_time = 10
    success_time = 0
    for i in range(test_time):
        done = False
        state_dic = env.reset(isTest=True)
        observation = dic_to_list(state_dic)
        reward_this_epi = 0
        while not done:
            # env.render()
            action = agent_test.choose_action(
                observation, episode, isTrain=False)
            state_dic, reward, done, truncated, _ = env.step(action)
            # print(reward)

            observation_ = dic_to_list(state_dic)
            # print(action,observation_,reward)
            reward = torch.tensor(
                reward, dtype=torch.long, device=device)
            reward_this_epi = reward_this_epi + reward.item()
            if done or truncated:
                if reward == 500:
                    success_time += 1
                    if success_time/test_time >= 0.8:
                        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with open(filename, "a") as file:
                            file.write(f"Time: {current_time}\n")
                            file.write(f"episoded: {episode}\n")

                break
            else:

                observation = observation_
    env.close()


if __name__ == '__main__':
    print(Back.YELLOW + Fore.BLACK +
          "---------------------------------------Training :---------------------------------------")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(Fore.RED + 'CUDA是否可用:',  torch.cuda.is_available())

    print(Fore.RED + 'CUDA设备数量:',  torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(Fore.RED + f'设备{i}名称:',  torch.cuda.get_device_name(i))
    main()
