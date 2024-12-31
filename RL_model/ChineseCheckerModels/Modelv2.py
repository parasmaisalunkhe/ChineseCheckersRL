import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.spaces import MultiDiscrete, Box
from enum import Enum

class Actions(Enum):
  RIGHT = 0
  UPRIGHT = 1
  UPLEFT = 2
  DOWNLEFT = 3
  DOWNRIGHT = 4

class ChineseCheckersEnv(gymnasium.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human', 'ascii']}
  def ChineseCheckersPattern(width, height):
    finalpattern = "." * width
      # holes = [1,2,3,4,13,12,11,10,9,10,11,12,13,4,3,2,1]
    holes = [1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1]
    for n in holes:
        pattern = ""
        for i in range(n):
            pattern += "x."
        pattern = pattern[:-1]
        while len(pattern) != width:
          pattern = "." + pattern + "."
        finalpattern += pattern
    finalpattern += "." * width
    return np.array([0 if char == 'x' else -1 for char in finalpattern], dtype=np.uint8)#.reshape((height, width, 3))
  def __init__(self):
    super(ChineseCheckersEnv, self).__init__()
    self.cumulative_reward = 0
    self.defaultPlayers = ["Red", "Blue"]
    self.height, self.width = 17, 27
    self.state = self.ChineseCheckersPattern()
    self.action_space = MultiDiscrete([self.height*self.width, 12])
    self.observation_space = Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
    self.mypieces = 
  def step(self, action):
    # action is a tuple of (piece, hole)
    if action[1] 

    ...
    return observation, reward, done, truncated, info
  def reset(self):
    self.state = self.ChineseCheckersPattern()

    ...
    return observation  # reward, done, info can't be included
  def render(self, mode='human'):
    plt.imshow(self.state.reshape(self.height, self.width).squeeze())
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()
  def close (self):
    plt.close('all')