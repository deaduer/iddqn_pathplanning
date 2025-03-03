# iddqn 
import datetime
import argparse
import gymnasium as gym
from tqdm import tqdm
import torch
import torch.nn.functional as F
from DDQN import DDQN
from utils import plot_learning_curve, create_directory
from colorama import Fore, Back,  init
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

tau = 0.05 

epsilon = 1.0  
eps_end = 0.05 

eps_dec = 500  
max_size = 10000  
batch_size = 64 
n_multi_step = 3  
parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=num_episodes)
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/DDQN/')
parser.add_argument('--reward_path', type=str,
                    default='./output_images/avg_reward.png')
parser.add_argument('--epsilon_path', type=str,
                    default='./output_images/epsilon.png')
args = parser.parse_args()




env = gym.make(env_name)
env.reset()
state_dic, reward, done, truncated, info = env.step(
    env.action_space.sample())
n_actions = env.action_space.n
state_dim = len(state_dic)
env.close()





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
                observation =state_dic
                while not done:

                    action = agent.choose_action(
                        observation, episode, isTrain=True)
                    state_dic, reward, done, _, _ = env.step(action)
                    observation_ = state_dic
                    agent.remember(
                        observation[-1], action, reward, observation_[-1], done)
                    agent.learn()
                    total_reward += reward
                    observation = observation_
                pbar.update(1)

            



eval_name = '\\Q_eval\\DDQN_q_eval_'  # 请替换为您的测试名称
target_name = '\\Q_target\\DDQN_Q_target_'  # 请替换为您的目标名称






if __name__ == '__main__':
    print(Back.YELLOW + Fore.BLACK +
          "---------------------------------------Training :---------------------------------------")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
