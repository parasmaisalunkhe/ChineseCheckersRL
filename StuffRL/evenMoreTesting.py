import gym
import numpy as np
import random
from GymEnv import ChineseCheckersBoard
import time
# Create the environment (replace with your custom env if needed)
env = ChineseCheckersBoard(2)  # or just use your env class directly: env = YourEnv()

obs, info = env.reset()
done = False
total_reward = 0
env.render(env.GlobalBoard)
while not done:
    mask = obs["action_mask"]
    stacked = np.stack(mask)

# Get index of the max 2nd element
    idx = np.argmax(stacked[:, 1])

    # Return the original array from the list
    result = mask[idx]

    # print("mask:", mask)
    action =  result #random.choice(mask)
    obs, reward, done, truncations, info = env.step(action)
    env.render(env.GlobalBoard)
    total_reward += reward
    time.sleep(1)
    print("\033[H\033[J")
    
    # print(f"Action: {action}, Reward: {reward}, Done: {done}")

print("Episode finished with total reward:", total_reward)
