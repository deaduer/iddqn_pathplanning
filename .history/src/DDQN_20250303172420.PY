import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
import math

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


# class DeepQNetwork(nn.Module):
#     def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim, fc3_dim):
#         super(DeepQNetwork, self).__init__()

#         self.fc1 = nn.Linear(state_dim, fc1_dim)
#         self.fc2 = nn.Linear(fc1_dim, fc2_dim)
#         self.fc3 = nn.Linear(fc2_dim, fc3_dim)  # 新增隐藏层
#         self.q = nn.Linear(fc3_dim, action_dim)

#         self.optimizer = optim.Adam(self.parameters(), lr=alpha)
#         self.to(device)

#     def forward(self, state):
#         x = T.relu(self.fc1(state))
#         x = T.relu(self.fc2(x))
#         x = T.relu(self.fc3(x))  # 新增隐藏层的前向传播
#         q = self.q(x)

#         return q
class DeepQNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(DeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        # dropout_rate = 0.5
        # self.dropout1 = nn.Dropout(dropout_rate)  # 添加dropout层
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, state):
        x = T.relu(self.fc1(state))
        # x = self.dropout1(x)  # 使用dropout层#TODO删除
        x = T.relu(self.fc2(x))

        q = self.q(x)

        return q

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file,
               _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))

# TODO 引入自己写的贪心策略


class DDQN:
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim, ckpt_dir,
                 gamma=0.99, tau=0.005, epsilon=1.0, eps_end=0.01, eps_dec=5e-7,
                 max_size=10000, batch_size=256, n_multi_step=3, num_episodes=5000):
        # 折扣因子
        self.gamma = gamma
        self.n_multi_step = n_multi_step
        # 软更新系数
        self.tau = tau
        # 探索概率初始值
        self.epsilon = epsilon
        # 探索概率最小值
        self.eps_min = eps_end
        # 探索概率衰减率
        self.eps_dec = eps_dec
        # 批量大小
        self.batch_size = batch_size
        # 检查点目录
        self.checkpoint_dir = ckpt_dir
        # 动作空间
        self.action_space = [i for i in range(action_dim)]
        # 总迭代次数
        self.num_episodes = num_episodes

        # 初始化评估网络
        self.q_eval = DeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                   fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        # 初始化目标网络
        self.q_target = DeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=fc1_dim, fc2_dim=fc2_dim)

        # 初始化经验回放缓冲区
        self.memory = ReplayBuffer(state_dim=state_dim, action_dim=action_dim,
                                   max_size=max_size, batch_size=batch_size, n_multi_step=n_multi_step, gamma=gamma)

        # 使用tau=1.0初始化网络参数
        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(
                tau * q_eval_params + (1 - tau) * q_target_params)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation, episode, isTrain=True):
        self.episode = episode
        state = T.tensor(observation, dtype=T.float).to(device)
        actions = self.q_eval.forward(state)
        action = T.argmax(actions[-1]).item()
        if (np.random.random() < self.epsilon) and isTrain:  # 引入自己写的贪心策略
            action = np.random.choice(self.action_space)
        return action

    def decrement_epsilon(self):  # TODO引入自己写的贪心策略
        u1 = 0.99
        u2 = 0.1
        self.epsilon = self.eps_min + \
            (self.epsilon-self.eps_min) / \
            (u1+math.exp(self.episode / (u2*self.num_episodes)))

    def learn(self):
        if not self.memory.ready():
            return

        states, actions, rewards, next_states, terminals = self.memory.sample_buffer()
        batch_idx = np.arange(self.batch_size)

        states_tensor = T.tensor(states, dtype=T.float).to(device)
        rewards_tensor = T.tensor(rewards, dtype=T.float).to(device)
        next_states_tensor = T.tensor(next_states, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)

        with T.no_grad():
            '''
            ddqn网络 计算target_q值时用两个网络计算[评估网络和target网络]
            dqn网络  计算target_q值时用一个网络计算[target网络]
            
            '''
            q_ = self.q_eval.forward(next_states_tensor)
            next_actions = T.argmax(q_, dim=-1)
            q_ = self.q_target.forward(next_states_tensor)
            q_[terminals_tensor] = 0.0
            # target = rewards_tensor + (self.gamma **self.n_multi_step)* q_[batch_idx, next_actions]#TODOn_step
            target = rewards_tensor + self.gamma * q_[batch_idx, next_actions]
        q = self.q_eval.forward(states_tensor)[batch_idx, actions]

        loss = F.mse_loss(q, target.detach())
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()

        self.update_network_parameters()
        self.decrement_epsilon()
        return loss.item()

    def save_models(self, episode):
        self.q_eval.save_checkpoint(
            self.checkpoint_dir + 'Q_eval/DDQN_q_eval_{}.pth'.format(episode))
        # print('Saving Q_eval network successfully!')
        self.q_target.save_checkpoint(
            self.checkpoint_dir + 'Q_target/DDQN_Q_target_{}.pth'.format(episode))
 

    def load_models(self, eval_path, target_path):  # episode)

        self.q_eval.load_checkpoint(eval_path)

        self.q_target.load_checkpoint(target_path)
        print('\nLoading network successfully!')
