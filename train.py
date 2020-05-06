''' 문제점 발생시 다음을 시도해볼 것
        pip install -U pillow
        pip install nes-py : 해당 패키지 설치시 문제가 발생할 경우, visual studio community c++를 설치하면 해결된다.
'''
import numpy as np
import math
import matplotlib.pyplot as plt
# import cPickle as pickle

# from JSAnimation.IPython_display import display_animation
# from matplotlib import animation
from collections import deque
# nes_py는 simpleNES(Nintendo Entertainment System) emulator를 기반으로하는 NES 에뮬레이터이며,
# openAI의 gym interface입니다.
from nes_py.wrappers import JoypadSpace
# super mairo 환경을 만들기 위해서는 꼭 gym_super_mario_bros를 import해야합니다.
# gym_super_mario_bros 환경은 256의 모든 NES action space actions를 사용합니다
import gym_super_mario_bros
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv

# NES action에 제약을 걸기 위해서는, gym_super_mario_bros.actions을 사용해야합니다.
# 먼저 ~.actions는 아래의 3가지 action list를 제공합니다.
# RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
# 위 세가지는 nes_py.wrappers.JoypadSpace의 wrapper로 사용됩니다.
# 그 중에서 여기는 SIMPLE_MOVEMENT를 사용하기로 하였습니다.

# super_mario_bros 환경 만들기
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
# action_space 재정의(nes-py)
# env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = JoypadSpace(env, COMPLEX_MOVEMENT)
# observation_space 재정의(gym)
env = PreprocessFrame(env)
# life cycle 재정의
# env = EpisodicLifeEnv(env)
# reward 재정의 필요


episode = 3000
INITIAL_BUFFER_SIZE = 1000  # 초기 inital 값을 버퍼에 채우기 전까지는 학습하지 않습니다.
# 최소의 explore를 하기 위한 value
EPS = 1.00  # 3 프로의 확률로 랜덤
EPS_THRESHOLD = 0.01  # 무한번 반복 시켜도 1%의 최소 확률을 남겨놓음
EPS_DECAY = 0.99  # 0.99를 곱한다.

# keep track of progress
# episode가 진행됨에 따라, average score가 어떻게 변화하는지 표로 보여주기 위해
sum_rewards = []
losses = []
# keep track of frames
FRAME_SHAPE = (84, 84)
MAX_FRAMES = 4
# (4 * 84 * 84)의 size의 states initialization
nn_frames = deque(maxlen=MAX_FRAMES)
for i in range(MAX_FRAMES):
    nn_frames.append(np.zeros(FRAME_SHAPE))

action_size = env.action_space.n
state_size = (MAX_FRAMES,) + FRAME_SHAPE
# aget 객체 생성시, 네트워크 종류를 꼭 선택해야 합니다. ddqn이 false인거 항상 확인
agent = DQNAgent(CNN_DDQN, state_size, action_size, INITIAL_BUFFER_SIZE, ddqn=True, priority=True)

for e in range(episode):
    # 매 episode 시작마다 environment를 reset합니다.
    obs = env.reset()
    frame_number = 0
    sum_reward = 0
    prev_action = 0
    # 일정 counter가 찰때까지 화면이 멈춰있으면 패널티를 추가합니다.
    # freeze_counter = 0
    # Replay_Buffer에 고정된 상황만 쌓이는걸 방지하기 위한 것입니다.
    buffer_safety_counter = 0
    buffer_safety_counter_limit = 100

    for i in range(MAX_FRAMES):
        nn_frames.append(np.zeros(FRAME_SHAPE))
    nn_frames.append(np.copy(obs))
    states = np.array(nn_frames)

    while True:
        # Skipped frame적용. 매 4 frame마다.
        # '''
        if frame_number % 4 != 0:
            _, reward, done, _ = env.step(prev_action)
            if done:
                break
            # 오버플로우 위험
            frame_number += 1
            # print(frame_number)
            sum_reward += reward
            # env.render()
            continue
        # '''
        actions = agent.act(states, EPS)
        obs, reward, done, info = env.step(actions.item())
        # action을 취한 후 받은 obs를 nn_frames에 담는다. 3차원에서 2차원으로
        nn_frames.append(np.copy(obs))
        next_states = np.array(nn_frames)
        # print(sum_reward)
        # print(reward)
        # ReplayBuffer에 정체된 상황만 쌓이는 것을 방지하기 위한 것입니다.
        if reward == 0 or reward == -1:
            buffer_safety_counter += 1
        else:
            buffer_safety_counter = 0
        # print(reward)
        # 받은 obs를 리플레이 메모리에 저장합니다.
        # 또한 INITIAL_BUFFER_SIZE 이상이 되면, 네트워크를 update합니다.
        agent.step(states, int(actions), int(reward), next_states, int(done))
        # 한 episode에서 최종 reward를 계산하기 위해
        sum_reward += reward
        # 다음 state를 여기서 왜 준비할까?
        states = next_states
        # 끝나는 지점 표시
        if done or buffer_safety_counter == buffer_safety_counter_limit or reward < -10:
            # buffer_safety_counter = 0
            # reward -= 100
            break
        # '''
        frame_number += 1
        prev_action = actions
        # '''
        # env.render()

    # if len(agent.memory) >= INITIAL_BUFFER_SIZE:
    # 하나의 episode가 끝나면, error를 업데이트한다.
    # priority update를 위해서
    agent.update_error()
    # if e % 10 == 0:
    # agent.soft_update(agent.qnetwork_local, agent.qnetwork_target, TAU)
    # agent.qnetwork_local.load_state_dict(agent.qnetwork_target.state_dict())
    # get the average reward of the environments
    sum_rewards.append(sum_reward)
    # 매 episode마다 nose가 0.05에서 감소한다. 99%에서 매번 0.99를 곱하여 감소시킨다)
    if buffer_safety_counter == buffer_safety_counter_limit:
        buffer_safety_counter = 0
        # EPS += 0.0001
    else:
        EPS = EPS_THRESHOLD + (EPS - EPS_THRESHOLD) * EPS_DECAY
        # EPS = EPS_THRESHOLD + (EPS - EPS_THRESHOLD) * math.exp(-1. * e / episode)
        # print(EPS)
    print('\rEpisode {}\tCurrent Score: {:.2f}'.format(e, sum_rewards[-1]), end="")
    # display some progress every 20 iterations
    print(" | Episode: {0:d}, average score: {1:f}".format(e + 1, np.mean(sum_rewards[-20:])), end="")
    print(" | EPS: {0:f} ".format(EPS))

# 자동으로 종료하고 메모리를 반환다고 하지만 작성하였다.
env.close()
# plot(e, sum_rewards, losses[0])
plt.plot(sum_rewards)
plt.show()
# '''