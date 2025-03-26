# from Model import ChineseCheckersEnv
from Modelv2 import ChineseCheckersEnvV2
import cv2
# from stable_baselines3 import PPO
Players = ["B", "G"]
env = ChineseCheckersEnvV2()
# print(env.observation_space.sample())
# print(env.action_space.sample())

episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0
    while not done:
        temp = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(temp)
        env.render()
        score+=reward
    cv2.destroyAllWindows()
    print('Episode:{} Score:{}'.format(episode, score))


# model = PPO.load("Third Attempt.zip")
# obs = env.reset()
# obs = obs[0]
# print(obs)
# done = False
# while not done:
#     action, _states = model.predict(obs)
#     obs, reward, done, _, info = env.step(action)
#     env.render()
# env.close()
# model = PPO("MlpPolicy", env, dverbose=1)
# model.learn(total_timesteps=1000000)
# model.save("Third Attempt")

# from stable_baselines3.common.env_checker import check_env
# check_env(ChineseCheckersEnv, warn=True)