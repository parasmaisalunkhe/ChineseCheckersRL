import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from TestingGymEnvMore import ChineseCheckersBoard
import random

env = ChineseCheckersBoard(n_players=2)

# Parameters
num_episodes = 500
alpha = 0.1         # Learning rate
gamma = 0.99        # Discount factor

# Initialize state-value function
V = defaultdict(float)

losses = []

def state_to_key(obs):
    return tuple(obs["obs"].astype(int))  # Convert to hashable key

for ep in range(num_episodes):
    obs, _ = env.reset()
    done = False
    total_loss = 0
    print(ep)
    while not done and env.num_moves < 1000:
        state_key = state_to_key(obs)
        # Choose a random valid action
        legal_moves = obs["action_mask"]
        action = random.choice(legal_moves)

        next_obs, reward, done, _, _ = env.step(action)
        next_state_key = state_to_key(next_obs)

        # TD(0) Update
        td_target = reward + gamma * V[next_state_key]
        td_error = td_target - V[state_key]
        V[state_key] += alpha * td_error

        total_loss += td_error**2
        obs = next_obs   

    losses.append(total_loss)

# Plotting
plt.plot(losses)
plt.xlabel("Episode")
plt.ylabel("TD Loss")
plt.title("TD(0) Loss vs. Episode")
plt.grid(True)
plt.show()
