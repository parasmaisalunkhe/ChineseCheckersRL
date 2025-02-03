import gymnasium
import gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.spaces import MultiDiscrete, Box
# from stable_baselines3.common.env_checker import check_env
class ChineseCheckersEnvV2(gymnasium.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human', 'ascii']}
  def ChineseCheckersPattern(self):
    width = 19
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
        # print(pattern)
    finalpattern += "." * width
    temp = np.array([0 if char == 'x' else -1 for char in finalpattern], dtype=int).reshape(width, width)
    return temp
  def __init__(self):
    super(ChineseCheckersEnvV2, self).__init__()
    self.cumulative_reward = 0
    self.defaultPlayers = ["Red", "Blue"]
    self.done = False
    self.height, self.width = 19, 19
    self.state = self.ChineseCheckersPattern()
    self.action_space = MultiDiscrete([self.height*self.width, 12])
    self.observation_space = Box(low=-1, high=2, shape=(self.height,self.width), dtype=int)
    allOcc = np.where(self.state == 0)
    self.goalpieces = list(zip(allOcc[0], allOcc[1]))[:10]
    self.mypieces = list(zip(allOcc[0], allOcc[1]))[-10:]
    # print(self.goalpieces, self.mypieces)
    for point in self.mypieces:
      print(point)
      self.state[point[0]][point[1]] = 1
    for point in self.goalpieces:
      self.state[point[0]][point[1]] = 2
    print(self.state)
  def step(self, action):
    # self.state[np.random.choice(np.where(self.state == -1)[0], size=int((self.state == -1).sum() * 0.5), replace=False)] = 2
    print(self.height, self.width, self.height * self.width)
    print("action:", action)
    index = action[0]
    row = index // self.width
    col = index % self.width
    # RIGHT = 0
    # UPRIGHT = 1
    # UPLEFT = 2
    # DOWNLEFT = 3
    # DOWNRIGHT = 4
    # LEFT = 5
    # JUMPRIGHT = 6
    # JUMPLEFT = 7
    # JUMPUPRIGHT = 8
    # JUMPUPLEFT = 9
    # JUMPDOWNLEFT = 10
    # JUMPDOWNRIGHT = 11
    print(row, col)
    self.state[row][col] = -1
    if action[1] == 0:
      self.state[row+1][col] = 1
    elif action[1] == 1:
      self.state[row+1][col+1] = 1
    elif action[1] == 2:
      self.state[row-1][col+1] = 1
    elif action[1] == 3:
      self.state[row-1][col-1] = 1
    elif action[1] == 4:
      self.state[row+1][col-1] = 1
    elif action[1] == 5:
      self.state[row-1][col] = 1
    elif action[1] == 6:
      self.state[row+2][col] = 1
    elif action[1] == 7:
      self.state[row-2][col] = 1
    elif action[1] == 8:
      self.state[row+2][col+2] = 1
    elif action[1] == 9:
      self.state[row-2][col+2] = 1
    elif action[1] == 10:
      self.state[row-2][col-2] = 1
    elif action[1] == 11:
      self.state[row+2][col-2] = 1
    self.state = self.state.flatten()
    if self.isDone():
      return self.state, 10, True, False, None
    return self.state, -0.1, False, False, None
  def reset(self, seed=None):
    self.state = self.ChineseCheckersPattern()
    self.cumulative_reward = 0
    return (self.state, dict())  # reward, done, info can't be included
  def render(self, mode='human'):
    plt.imshow(self.state.reshape(self.height, self.width).squeeze())
    plt.axis('off')
    plt.show()
  def close(self):
    plt.close('all')
  def isDone(self):
    for x in self.goalpieces:
      if self.state[x] == 0:
        return False
    return True
