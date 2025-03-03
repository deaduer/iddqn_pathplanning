# TODO 改为n-step
# TODO 把gamma改入参数中
import numpy as np
import random


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, batch_size, n_multi_step, gamma):
        self.gamma = gamma
        self.mem_size = max_size
        self.batch_size = batch_size
        self.mem_cnt = 0
        self.n_multi_step = n_multi_step  # n-step
        self.state_dim = state_dim
        self.state_memory = np.zeros((self.mem_size, state_dim))
        self.action_memory = np.zeros((self.mem_size, ))
        self.reward_memory = np.zeros((self.mem_size, ))
        self.next_state_memory = np.zeros((self.mem_size, state_dim))
        self.terminal_memory = np.zeros((self.mem_size, ), dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        mem_idx = self.mem_cnt % self.mem_size

        self.state_memory[mem_idx] = state  # 前一状态
        self.action_memory[mem_idx] = action
        self.reward_memory[mem_idx] = reward
        self.next_state_memory[mem_idx] = state_  # 后一状态
        self.terminal_memory[mem_idx] = done

        self.mem_cnt += 1

    # def sample_buffer(self):#N_STEP 采样
    #     actions, rewards,  dones = [], [], []
    #     states, next_states = np.zeros((self.batch_size, self.state_dim)), np.zeros(
    #         (self.batch_size, self.state_dim))
    #     # states, next_states = np.zeros(
    #         # (self.batch_size, 2)), np.zeros((self.batch_size, 2))
    #     mem_len = min(self.mem_size, self.mem_cnt)
    #     for i in range(self.batch_size):
    #         finish = random.randint(self.n_multi_step, mem_len - 1)
    #         begin=finish-self.n_multi_step
    #         sum_reward=0#n-step奖励之和
    #         data_state = self.state_memory[begin:finish]
    #         data_action=self.action_memory[begin:finish]
    #         data_reward=self.reward_memory[begin:finish]
    #         data_dones=self.terminal_memory[begin:finish]
    #         data_next_state = self.next_state_memory[begin:finish]
    #         state=data_state[0]
    #         action=data_action[0]
    #         for j in range(self.n_multi_step):
    #             # compute the n-th reward
    #             sum_reward += (self.gamma**j) * data_reward[j]
    #             if data_dones[j]:
    #                 states_look_ahead=data_next_state[j]
    #                 done_look_ahead=True
    #                 break
    #             else:
    #                 states_look_ahead=data_next_state[j]
    #                 done_look_ahead=False
    #         states[i]=state.tolist()
    #         actions.append(action)
    #         rewards.append(sum_reward)
    #         next_states[i]=states_look_ahead.tolist()
    #         dones.append(done_look_ahead)
    #     actions=np.array(actions)
    #     rewards=np.array(rewards)
    #     dones=np.array(dones)
    #     return states, actions, rewards, next_states, dones
    def sample_buffer(self):
        mem_len = min(self.mem_size, self.mem_cnt)

        batch = np.random.choice(
            mem_len, self.batch_size, replace=False)  # 随机采样

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminals

    def ready(self):
        return self.mem_cnt > self.batch_size
