import gymnasium
import gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.spaces import MultiDiscrete, Box
from stable_baselines3.common.env_checker import check_env
class ChineseCheckersEnv(gymnasium.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human', 'ascii']}
  def ChineseCheckersPattern(width):
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
    finalpattern += "." * width
    return np.array([0 if char == 'x' else -1 for char in finalpattern], dtype=int)#.reshape((height, width, 3))
  def __init__(self):
    super(ChineseCheckersEnv, self).__init__()
    self.cumulative_reward = 0
    self.defaultPlayers = ["Red", "Blue"]
    self.done = False
    self.height, self.width = 19, 19
    print(self.width)
    temp = self.ChineseCheckersPattern()
    self.state = temp
    self.action_space = MultiDiscrete([self.height*self.width, 12])
    self.observation_space = Box(low=-1, high=2, shape=(self.height * self.width,), dtype=int)
    self.goalpieces = np.where(self.state == 0)[0][:10].tolist()
    self.mypieces = np.where(self.state == 1)[0][-10:].tolist()
    for x in self.mypieces:
      self.state[x] = 1
  def step(self, action):
    self.state[np.random.choice(np.where(self.state == -1)[0], size=int((self.state == -1).sum() * 0.5), replace=False)] = 2
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
    grid = self.state.reshape(self.height, self.width)
    row, col = np.unravel_index(action, grid.shape)
    grid[row][col] = -1
    if action[1] == 0:
      grid[row+1][col] = 1
    elif action[1] == 1:
      grid[row+1][col+1] = 1
    elif action[1] == 2:
      grid[row-1][col+1] = 1
    elif action[1] == 3:
      grid[row-1][col-1] = 1
    elif action[1] == 4:
      grid[row+1][col-1] = 1
    elif action[1] == 5:
      grid[row-1][col] = 1
    elif action[1] == 6:
      grid[row+2][col] = 1
    elif action[1] == 7:
      grid[row-2][col] = 1
    elif action[1] == 8:
      grid[row+2][col+2] = 1
    elif action[1] == 9:
      grid[row-2][col+2] = 1
    elif action[1] == 10:
      grid[row-2][col-2] = 1
    elif action[1] == 11:
      grid[row+2][col-2] = 1
    self.state = grid.flatten()
    if self.isDone():
      return grid.flatten(), 10, True, dict(), None
    return grid.flatten(), -0.1, False, dict(), None
  def reset(self, seed=None):
    self.state = self.ChineseCheckersPattern()
    print(self.state.shape)
    self.cumulative_reward = 0
    return (self.state, dict())  # reward, done, info can't be included
  def render(self, mode='human'):
    plt.imshow(self.state.reshape(self.height, self.width).squeeze())
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()
  def close(self):
    plt.close('all')
  def isDone(self):
    for x in self.goalpieces:
      if self.state[x] == 0:
        return False
    return True
  def display(self):
    plt.imshow(self.state.squeeze())
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()
check_env(ChineseCheckersEnv(), warn=True)