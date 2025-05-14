from GymEnv import ChineseCheckersBoard
import numpy as np
import time
import cv2
import random

env = ChineseCheckersBoard(2)
num_steps_max = 10000000
obs, info = env.reset()

random_action_prob = 0.3  # 30% chance to take a random valid action

# def score(observation):
#     newObs = observation.reshape((2, 4))

while env.num_moves < num_steps_max:
    env.render(env.GlobalBoard)
    input()
    print("\033[H\033[J")
    lastobs = obs
    best_action = None
    best_score = float('inf')  # Lower is better
    best_obs = None

    current_player = env.current_player
    valid_actions = env.allLegalActions

    if not valid_actions:
        print("No valid actions, breaking.")
        break

    # With random probability, take a random valid action
    if random.random() < random_action_prob:
        best_action = random.choice(valid_actions)
        print("Taking random action:", best_action)
    else:
        for action in valid_actions:
            # Backup environment state
            backup_board = env.GlobalBoard.copy()
            backup_player = env.current_player
            backup_num_moves = env.num_moves

            try:
                temp_obs, reward, done, truncated, info = env.step(action)
                measurements = temp_obs  # [avg_x, avg_y, dist_to_corner]
                # print(temp_obs)
                result = temp_obs - lastobs

# Flatten the result
                flattened_result = result.flatten()

                # Calculate the squared sum and divide by 100
                squared_sum = np.sum(flattened_result)
                score = squared_sum
                print(score)
                # score = sum(measurements)
                
                if score < best_score:
                    best_score = score
                    best_action = action
                    best_obs = temp_obs

                # Restore environment state
                env.GlobalBoard = backup_board
                env.current_player = backup_player
                env.num_moves = backup_num_moves

            except Exception as e:
                print(f"Invalid action {action} caused exception: {e}")
                # Restore environment state
                env.GlobalBoard = backup_board
                env.current_player = backup_player
                env.num_moves = backup_num_moves
        # print(measurements)
    # Take the selected action
    lastobs = best_obs
    if best_action is not None:
        obs, reward, done, truncated, info = env.step(best_action)
    else:
        print("No valid action found, breaking.")
        break
