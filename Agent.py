import numpy as np
import random
from collections import namedtuple, deque
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(10000)  # replay buffer size
BATCH_SIZE = 64  # minibatch size  # 원래는 256
INITIAL_BUFFER_SIZE = 1000
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters 0.001
LR = 2e-5  # learning rate 0.0005
ETA = 0.001
UPDATE_EVERY = 10000  # how often to update the network 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent():
    """Interacts with and learns from the environment."""

    # Replay_Buffer와 Network를 초기화 하여 사용한다.
    def __init__(self, model, state_size, action_size, initial_buffer_size, seed=42, ddqn=False, priority=False):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        # sate_size = (84, 84, 4) 의 3차원의 크기를 가집니다.
        self.state_size = state_size
        # Preprocessing에서 SIMPLE_MOVEMETN의 갯수의 크기입니다.
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.ddqn = ddqn

        # Q-Network
        # model함수는 models 파일에서 가져온 모델 이름 class를 의미합니다.
        # parameter는 (channel_size, action_size, seed, w, h)
        # state_size[0]에는 frame_size가 들어갑니다.
        # model은 channel수만 신경쓰면 됩니다.
        self.qnetwork_local = model(state_size[0], action_size, seed).to(device)
        self.qnetwork_target = model(state_size[0], action_size, seed).to(device)
        # 최적화 방법 설정
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        # self.memory = ReplayBuffer(BUFFER_SIZE)
        self.memory = ReplayBuffer(state_size, (action_size,), BUFFER_SIZE, BATCH_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.initial_buffer_size = initial_buffer_size

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            # step 함수에 충분한 buffer가 차면 그때부터 학습을 시작합니다.
            if len(self.memory) > INITIAL_BUFFER_SIZE:  # 원래는 BATCH_SIZE였음
                # batch_size만큼 sampling후에 experiences와 idx로 나누어 저장합니다.
                experiences, idx = self.memory.sample()
                # e is error
                # 먼저 learn한 후에(error를 역전파)
                e = self.learn(experiences)
                # error를 replay buffer에 update합니다.
                self.memory.update_error(e, idx)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # unsqueeze()함수는 인수로 받은 위치에 새로운 차원을 삽입한다.(1, 4, 84, 84)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # 추론을 실행하기 전에는 반드시 model.eval() 을 호출하여 드롭아웃 및 배치 정규화를 평가 모드(dropout 사용안함)로 설정하여야 합니다. 이것을 하지 않으면 추론 결과가 일관성 없게 출력됩니다.
        # 평가시에는 연산 추적을 중단해야 메모리 효율적이고 정확하게 계산할 수 있습니다.
        self.qnetwork_local.eval()
        # 기록을 추적하는 것(과 메모리를 사용하는 것)을 방지하기 위해, 코드 블럭을 with torch.no_grad(): 로 감쌀 수 있습니다.
        # 이는 특히 변화도(gradient)는 필요없지만, requires_grad=True 가 설정되어 학습 가능한 매개변수를 갖는 모델을 평가(evaluate)할 때 유용합니다.
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        # train 시작전에 model.train()으로 dropout사용을 명시해줘야함
        # 여기를 왜 train 시작해주지?
        self.qnetwork_local.train()
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def update_error(self):

        states, actions, rewards, next_states, dones = self.memory.sample(get_all=True)
        # 기록을 추적하는 것(과 메모리를 사용하는 것)을 방지하기 위해, 코드 블럭을 with torch.no_grad(): 로 감쌀 수 있습니다.
        # 이는 특히 변화도(gradient)는 필요없지만, requires_grad=True 가 설정되어 학습 가능한 매개변수를 갖는 모델을 평가(evaluate)할 때 유용합니다.
        with torch.no_grad():
            if self.ddqn:
                old_val = self.qnetwork_local(states).gather(-1, actions)
                actions = self.qnetwork_local(next_states).argmax(-1, keepdim=True)
                maxQ = self.qnetwork_target(next_states).gather(-1, actions)
                target = rewards + GAMMA * maxQ * (1 - dones)
            else:  # Normal DQN
                maxQ = self.qnetwork_target(next_states).max(-1, keepdim=True)[0]
                target = rewards + GAMMA * maxQ * (1 - dones)
                old_val = self.qnetwork_local(states).gather(-1, actions)
            # priority를 위해서
            e = old_val - target
            self.memory.update_error(e)
        return e

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        # object 풀기 unpack experiences
        states, actions, rewards, next_states, dones = experiences

        ## compute and minimize the loss
        # 역전파 단계를 실행하기 전에 변화도를 0으로 만듭니다.
        # 역전파 단계 전에, Optimizer 객체를 사용하여 (모델의 학습 가능한 가중치인)
        # 갱신할 변수들에 대한 모든 변화도를 0으로 만듭니다. 이렇게 하는 이유는
        # 기본적으로 .backward()를 호출할 때마다 변화도가 버퍼(buffer)에 (덮어쓰지 않고)
        # 누적되기 때문입니다. 더 자세한 내용은 torch.autograd.backward에 대한 문서를
        # 참조하세요.

        self.optimizer.zero_grad()
        # double dqn일때
        if self.ddqn:
            # q_value
            old_val = self.qnetwork_local(states).gather(-1, actions)
            with torch.no_grad():
                next_actions = self.qnetwork_local(next_states).argmax(-1, keepdim=True)
                maxQ = self.qnetwork_target(next_states).gather(-1, next_actions)
                target = rewards + GAMMA * maxQ * (1 - dones)
        else:  # Normal DQN
            with torch.no_grad():
                maxQ = self.qnetwork_target(next_states).max(-1, keepdim=True)[0]
                target = rewards + GAMMA * maxQ * (1 - dones)
            old_val = self.qnetwork_local(states).gather(-1, actions)
            # MSE 계산
        loss = F.mse_loss(old_val, target)
        # loss = F.smooth_l1_loss(old_val, target)
        # print('Cost: {:.6f}'.format(loss.item()))
        # print(loss)
        # 여기가 backward()인가? NO!! 아래에 있음

        # backward()를 호출하여 그라디언트를 자동으로 계산합니다. 이 텐서를 위한 그라디언트는
        # .grad 속성에 누적되어 저장됩니다.
        loss.backward()
        # Optimizer의 step 함수를 호출하면 매개변수가 갱신됩니다.
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        # 여기도 제곱해줘야 하는거 아냐?
        # return old_val - target
        return old_val - target

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)