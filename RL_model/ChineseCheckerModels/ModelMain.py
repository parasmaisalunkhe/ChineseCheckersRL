from RL_model.ChineseCheckerModels.Model import ChineseCheckersEnv
import cv2
from stable_baselines3 import PPO
Players = ["B", "G"]
env = ChineseCheckersEnv(arg=Players)
print(env.observation_space.sample())
print(env.action_space.sample())

# episodes = 1
# for episode in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0
#         # env.boardState = "B" * 121
#         # env.render() 
        
#     while not done:
            
#         # action = env.action_space.sample()
#         # print(action.type)
#         # print(action, env.players)
#         # print(env.boardState) 
#         # print(env.player, env.boardState, len(env.boardState))
#         temp = env.action_space.sample()
#         # print(temp)
#         # env.rende/r()
#         # cv2.waitKey(0)
#         obs, reward, done, truncated, info = env.step(temp)
#             # print(obs, reward, done)
#             # print(done)
            
#         score+=reward
#     cv2.destroyAllWindows()
#     print('Episode:{} Score:{}'.format(episode, score))

model = PPO.load("Third Attempt.zip")
obs = env.reset()
obs = obs[0]
print(obs)
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    env.render()
env.close()

# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=1000000)
# model.save("Third Attempt")

# from stable_baselines3.common.env_checker import check_env
# check_env(env, warn=True)