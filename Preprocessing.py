import cv2
import gym
from gym import spaces
import numpy as np

# mario의 기본 observation space는 세로사이즈(height:224) 가로사이즈(width:256)
# spaces.Box(low=0, high=255, shape=(244,256,3))

#'A' 점프, 'B' 대쉬
SIMPLE_MOVEMENT = [
    ['right', 'B'],
    ['right', 'A'],
    ['right', 'A', 'B'],
]
COMPLEX_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'A'],
    ['right', 'A', 'B'],
    ['right', 'A', 'A','B']
]

ALL_MOVEMENT = [
    ['NOOP'],
    ['A'],
    ['B'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['down'],
    ['up']
]

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84,84))

    def observation(self, obs):
        return PreprocessFrame.process(obs)

    @staticmethod
    def process(img):
        img = img[:, :, 0]*0.299 + img[:, :, 1]*0.587 + img[:, :, 2]*0.114
        x_t = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        x_t = np.nan_to_num(x_t)
        # 이미지의 평균을 뺍니다.
        x_t = x_t - np.mean(x_t)
        return x_t.astype(np.int8)
