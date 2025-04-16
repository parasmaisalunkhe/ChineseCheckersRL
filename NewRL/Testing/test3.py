import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
import numpy as np

# Define your custom env class by subclassing gymnasium.Env:

class ParrotEnv(gym.Env):
    """Environment in which the agent learns to repeat the seen observations.

    Observations are float numbers indicating the to-be-repeated values,
    e.g. -1.0, 5.1, or 3.2.
    The action space is the same as the observation space.
    Rewards are `r=-abs([observation] - [action])`, for all steps.
    """
    def __init__(self, config=None):
        # Since actions should repeat observations, their spaces must be the same.
        self.observation_space = config.get(
            "obs_act_space",
            gym.spaces.Box(-1.0, 1.0, (1,), np.float32),
        )
        self.action_space = self.observation_space
        self._cur_obs = None
        self._episode_len = 0

    def reset(self, *, seed=None, options=None):
        """Resets the environment, starting a new episode."""
        # Reset the episode len.
        self._episode_len = 0
        # Sample a random number from our observation space.
        self._cur_obs = self.observation_space.sample()
        # Return initial observation.
        return self._cur_obs, {}

    def step(self, action):
        """Takes a single step in the episode given `action`."""
        # Set `terminated` and `truncated` flags to True after 10 steps.
        self._episode_len += 1
        terminated = truncated = self._episode_len >= 10
        # Compute the reward: `r = -abs([obs] - [action])`
        reward = -sum(abs(self._cur_obs - action))
        # Set a new observation (random sample).
        self._cur_obs = self.observation_space.sample()
        return self._cur_obs, reward, terminated, truncated, {}

# Point your config to your custom env class:
config = (
    PPOConfig()
    .environment(
        ParrotEnv,
        # Add `env_config={"obs_act_space": [some Box space]}` to customize.
    )
)

# Build a PPO algorithm and train it.
ppo_w_custom_env = config.build()
ppo_w_custom_env.training()