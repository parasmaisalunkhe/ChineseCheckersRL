from TestingGymEnvMore import ChineseCheckersBoard
import numpy as np
import time
import cv2
# import random

env = ChineseCheckersBoard(2)
num_steps_max = 10000000
obs, info = env.reset()

random_action_prob = 0.3  # 10% chance to take a random valid action

while env.num_moves < num_steps_max:
    env.render(env.GlobalBoard)
    input()
    print("\033[H\033[J")

    best_action = None
    best_score = float('inf')  # Lower is better
    best_obs = None

    current_player = env.current_player
    valid_actions = list(obs["action_mask"])

    # Decide whether to take a random action
    for action in valid_actions:
            # Backup environment state
            backup_board = env.GlobalBoard.copy()
            backup_player = env.current_player
            backup_num_moves = env.num_moves

            try:
                temp_obs, reward, done, truncated, info = env.step(action)
                measurements = temp_obs["measurements"]  # [avg_x, avg_y, dist_to_corner]
                score = sum(measurements)
                print(action, score)
                if score < best_score:
                    print("Best:", action, score)
                    best_score = score
                    best_action = action
                    best_obs = temp_obs
                # print("\033[H\033[J")
                # Restore environment state
                env.GlobalBoard = backup_board
                env.current_player = backup_player
                env.num_moves = backup_num_moves

            except Exception as e:
                print(f"Invalid action {action} caused exception: {e}")
                env.GlobalBoard = backup_board
                env.current_player = backup_player
                env.num_moves = backup_num_moves

    # Take the best or random action
    if best_action is not None:
        obs, reward, done, truncated, info = env.step(best_action)
    else:
        print("No valid action found, breaking.")
        break
