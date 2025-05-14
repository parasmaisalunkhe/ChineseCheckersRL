# from TestingGymEnvMore import ChineseCheckersBoard
# import numpy as np
# import random
# import nevergrad as ng

# # Balance weights: increase alpha to emphasize reward, beta to emphasize score
# alpha = 1.0  # how much we care about maximizing reward
# beta = 1.0   # how much we care about minimizing inefficiency score

# def evaluate_weights(weights, num_episodes=1):
#     total_reward = 0
#     total_score = 0

#     for _ in range(num_episodes):
#         env = ChineseCheckersBoard(2)
#         obs, info = env.reset()
#         episode_reward = 0
#         episode_score = 0

#         while not env.isGameOver(env.GlobalBoard, env.current_player):
#             valid_actions = list(obs["action_mask"])
#             best_action = None
#             best_score = float('inf')
#             best_measurement_score = float('inf')

#             for action in valid_actions:
#                 # Backup state
#                 backup_board = env.GlobalBoard.copy()
#                 backup_player = env.current_player
#                 backup_num_moves = env.num_moves

#                 try:
#                     temp_obs, reward, done, truncated, info = env.step(action)
#                     measurements = temp_obs["measurements"]  # [avg_x, avg_y, dist_to_corner]
#                     measurement_score = sum(w * m for w, m in zip(weights, measurements))

#                     if measurement_score < best_score:
#                         best_score = measurement_score
#                         best_action = action

#                     # Restore state
#                     env.GlobalBoard = backup_board
#                     env.current_player = backup_player
#                     env.num_moves = backup_num_moves

#                 except:
#                     env.GlobalBoard = backup_board
#                     env.current_player = backup_player
#                     env.num_moves = backup_num_moves

#             # Take the best action
#             obs, reward, done, truncated, info = env.step(best_action)
#             episode_reward += reward
#             episode_score += best_score

#         total_reward += episode_reward
#         total_score += episode_score

#     avg_reward = total_reward / num_episodes
#     avg_score = total_score / num_episodes

#     # We negate reward because optimizer minimizes
#     return alpha * (-avg_reward) + beta * avg_score

# # Nevergrad optimization wrapper
# def objective(weights_array):
#     return evaluate_weights(weights_array)

# # Optimizer setup
# optimizer = ng.optimizers.OnePlusOne(parametrization=3, budget=5)  # 3 weights, 20 evaluations
# recommendation = optimizer.minimize(objective)

# print("✅ Best weights found:", recommendation.value)
# print("🏆 Final score (combined objective):", recommendation.loss)

