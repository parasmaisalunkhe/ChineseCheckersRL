
from TestingGymEnvMore import ChineseCheckersBoard
import random
import time
import cv2
env = ChineseCheckersBoard(2)
num_steps_max = 1000
obs, info = env.reset()
done = False
while env.num_moves < num_steps_max and not done:
    env.render(env.GlobalBoard)
    print(obs["action_mask"])
    print(obs["obs"])
    print()
    print(obs["measurements"])
    action = random.choice(obs["action_mask"])
    obs, reward, done, truncated, info = env.step(action)
    print(done)
    input()
    print("\033[H\033[J")
# env.GlobalBoard = env.rotateNtimes(env.GlobalBoard, 3)
# print(env.isGameOver(env.GlobalBoard, env.current_player))