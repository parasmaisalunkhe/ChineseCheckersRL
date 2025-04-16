from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env

from chinesecheckers import ChineseCheckersBoard

register_env(
    "ChineseCheckers",
    lambda cfg: PettingZooEnv(ChineseCheckersBoard(6)),
)

config = (
    PPOConfig()
    .environment(env="ChineseCheckers")
)

algo = config.build()
print(algo.train())