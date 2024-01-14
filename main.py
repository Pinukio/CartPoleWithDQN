# 바닥부터 배우는 강화 학습 P.206 DQN 구현

import gym
# deque를 위해 import
# deque는 꽉 차면 자동으로 FIFO
import collections 
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

# 추측치를 학습하는 네트워크
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        # FC Layer를 사용
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        # 선택할 수 있는 Action이 2개
        self.fc3 = nn.Linear(128, 2)

    # State만 받아서 Action 2개의 Value를 모두 리턴
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 출력층은 Identity Function
        # Action Value는 음수도 가능한데, ReLU는 양수만 뱉기 때문
        x = self.fc3(x)
        return x
    
    def sample_action(self, obs, epsilon):
        coin = random.random()
        if coin < epsilon: # 랜덤한 Action을 취하는 경우
            return random.randint(0, 1)
        else: # Value가 최대인 Action을 취하는 경우
            out = self.forward(obs)
            # Value가 최대인 Action을 리턴
            return out.argmax().item()
    
def train(q, q_target, memory, optimizer):
    for i in range(10):
        # memory: buffer
        state, action, reward, s_prime, done_mask = memory.sample(batch_size)

        q_out = q.forward(state)
        # 추측치
        q_a = q_out.gather(1, action)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        # 정답값
        target = reward + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        # autoGrad 캐시 영역 초기화?
        optimizer.zero_grad()
        # loss에 대한 Gradient 계산
        loss.backward()
        # 계산된 Optimizer로 Gradient Descent
        optimizer.step()
        
