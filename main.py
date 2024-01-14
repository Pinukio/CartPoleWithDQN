# 바닥부터 배우는 강화 학습 P.206 DQN 구현

import gym
import collections # deque를 위해 import
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# HyperParameters
learning_rate = 0.0005
gamma = 0.98 # discount factor
buffer_limit = 50000 # repaly buffer 개수
batch_size = 32

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    # buffer에 경험을 넣음
    def put(self, transition):
        self.buffer.append(transition)

    # buffer에서 mini-batch를 랜덤하게 Sampling
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        # transition을 각 list로 분리하기 위함
        # done_mask는 게임이 실행 중일 때 1, 종료되었을 때 0임.
        state_list, action_list, reward_list, s_prime_list, done_mask_list = []
        
        for transition in mini_batch:
            state, action, reward, s_prime, done_mask = transition
            state_list.append(state)
            action_list.append([action])
            reward_list.append([reward])
            s_prime_list.append(s_prime)
            done_mask_list.append([done_mask])

        return torch.tensor(state_list, dtype=torch.float), torch.tensor(action_list), \
    torch.tensor(reward_list), torch.tensor(s_prime_list, dtype=torch.float), torch.tensor(done_mask_list)

    # buffer size를 리턴함.
    def size(self):
        return len(self.buffer)
    