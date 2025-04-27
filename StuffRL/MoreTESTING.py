import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from TestingGymEnvMore import ChineseCheckersBoard

# Hyperparameters
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
REPLAY_BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Outputs Q-value for (s,a) via measurement features
        )

    def forward(self, x):
        return self.net(x)

env = ChineseCheckersBoard(6)
q_net = QNetwork(3).to(device)
target_net = QNetwork(3).to(device)
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=LR)
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

epsilon = EPSILON_START
num_episodes = 1000
max_steps = 500

for episode in range(num_episodes):
    obs, info = env.reset()
    total_reward = 0

    for step in range(max_steps):
        if len(obs["action_mask"]) == 0:
            break

        measurements_list = []
        for action in obs["action_mask"]:
            # Simulate action to get resulting measurements
            backup_board = env.GlobalBoard.copy()
            backup_player = env.current_player
            backup_num_moves = env.num_moves

            try:
                temp_obs, reward, done, truncated, info = env.step(action)
                meas = temp_obs["measurements"]
                measurements_list.append((meas, action, reward, done))
                # Roll back
                env.GlobalBoard = backup_board
                env.current_player = backup_player
                env.num_moves = backup_num_moves
            except Exception:
                env.GlobalBoard = backup_board
                env.current_player = backup_player
                env.num_moves = backup_num_moves

        if not measurements_list:
            break

        # Epsilon-greedy selection
        if random.random() < epsilon:
            meas, action, reward, done = random.choice(measurements_list)
        else:
            scores = []
            for meas, action, reward, done in measurements_list:
                meas_tensor = torch.tensor(meas, dtype=torch.float32).to(device)
                q_value = q_net(meas_tensor).item()
                scores.append((q_value, meas, action, reward, done))
            _, meas, action, reward, done = min(scores, key=lambda x: -x[0])  # max Q-value

        # Take real step with best action
        next_obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Compute next measurement set for next Q-value prediction
        next_meas = []
        for next_action in next_obs["action_mask"]:
            try:
                backup_board = env.GlobalBoard.copy()
                backup_player = env.current_player
                backup_num_moves = env.num_moves
                temp_obs, _, _, _, _ = env.step(next_action)
                next_meas.append(temp_obs["measurements"])
                env.GlobalBoard = backup_board
                env.current_player = backup_player
                env.num_moves = backup_num_moves
            except:
                env.GlobalBoard = backup_board
                env.current_player = backup_player
                env.num_moves = backup_num_moves

        meas_tensor = torch.tensor(meas, dtype=torch.float32).to(device)
        reward_tensor = torch.tensor([reward], dtype=torch.float32).to(device)

        if done or not next_meas:
            target_q = reward_tensor
        else:
            next_qs = torch.stack([target_net(torch.tensor(m, dtype=torch.float32).to(device)) for m in next_meas])
            max_next_q = next_qs.max().detach()
            target_q = reward_tensor + GAMMA * max_next_q

        replay_buffer.append((meas_tensor, target_q))

        obs = next_obs

        # Learn from experience
        if len(replay_buffer) >= BATCH_SIZE:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            states, targets = zip(*batch)
            states = torch.stack(states)
            targets = torch.stack(targets).squeeze()

            pred_q = q_net(states).squeeze()
            loss = nn.MSELoss()(pred_q, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done or truncated:
            break

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(q_net.state_dict())

    print(f"Episode {episode}: Total reward = {total_reward:.2f}, Epsilon = {epsilon:.3f}")

