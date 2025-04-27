from GymEnv import ChineseCheckersBoard
import numpy as np
import time
import cv2

env = ChineseCheckersBoard(2)
num_steps_max = 10000000
obs, info = env.reset()

while env.num_moves < num_steps_max:

    # print(env.num_moves)
    env.render(env.GlobalBoard)
    input()
    print("\033[H\033[J")
    best_action = None
    best_score = float('inf')  # Lower is better
    best_obs = None

    current_player = env.current_player

    for action in obs["action_mask"]:
        # Clone env state to simulate the step safely
        backup_board = env.GlobalBoard.copy()
        backup_player = env.current_player
        backup_num_moves = env.num_moves

        try:
            temp_obs, reward, done, truncated, info = env.step(action)
            # print(reward)
            measurements = temp_obs["measurements"]  # [avg_x, avg_y, dist_to_corner]
            score = sum(measurements)  # You can weight this if needed

            if score < best_score:
                best_score = score
                best_action = action
                best_obs = temp_obs

            # Roll back environment to original state
            env.GlobalBoard = backup_board
            env.current_player = backup_player
            env.num_moves = backup_num_moves

        except Exception as e:
            print(f"Invalid action {action} caused exception: {e}")
            # Roll back anyway just in case
            env.GlobalBoard = backup_board
            env.current_player = backup_player
            env.num_moves = backup_num_moves

    # Take the best action found
    if best_action is not None:
        obs, reward, done, truncated, info = env.step(best_action)
    else:
        print("No valid action found, breaking.")
        break

    # 
# env.GlobalBoard = env.rotateNtimes(env.GlobalBoard, 3)
# print(env.isGameOver(env.GlobalBoard, env.current_player))