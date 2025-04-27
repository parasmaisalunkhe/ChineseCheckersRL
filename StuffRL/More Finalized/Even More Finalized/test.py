import gymnasium as gym
import numpy as np

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

# 1. The base environment
class SimpleMultiDiscreteEnv(gym.Env):
    """MultiDiscrete environment with masking."""
    
    def __init__(self):
        super().__init__()
        # Two separate discrete actions: one from [0,1,2], one from [0,1]
        self.action_space = gym.spaces.MultiDiscrete([3, 2])
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.state = None
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.rand(5).astype(np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action!"
        print("Action Taken:", action)
        # Reward +1 if first action == 2 and second action == 1
        reward = 1.0 if (action[0] == 2 and action[1] == 1) else -1.0
        done = self.steps >= 20
        self.steps += 1
        self.state = np.random.rand(5).astype(np.float32)

        info = {}
        return self.state, reward, done, False, info

    def valid_action_mask(self):
        """Return a flattened action mask for the flattened action space."""
        mask = np.zeros((3, 2), dtype=bool)

        # Define valid (first_action, second_action) pairs
        mask[2, 1] = True  # (2,1)
        mask[1, 0] = True  # (1,0)
        mask[0, 1] = True  # (0,1)

        # Flatten (row-major) to a 1D array
        return mask.flatten()


# 2. The flattening wrapper
class FlattenMultiDiscreteWrapper(gym.ActionWrapper):
    """Flattens MultiDiscrete into a single Discrete action space."""
    def __init__(self, env):
        super().__init__(env)
        nvec = env.action_space.nvec
        self.orig_action_space = env.action_space
        self.action_space = gym.spaces.Discrete(np.prod(nvec))
        self.nvec = nvec

    def action(self, action):
        """Convert flat action index into MultiDiscrete action array."""
        idxs = []
        for n in reversed(self.nvec):
            idxs.append(action % n)
            action //= n
        return np.array(list(reversed(idxs)), dtype=int)

    def valid_action_mask(self):
        """Forward the valid action mask from the original env."""
        return self.env.valid_action_mask()


# 3. Mask function for ActionMasker
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()


# 4. Instantiate and wrap the environment
env = SimpleMultiDiscreteEnv()
env = FlattenMultiDiscreteWrapper(env)  # Flatten actions
env = ActionMasker(env, mask_fn)         # Add action masking

# 5. Create and train the MaskablePPO model
model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
model.learn(total_timesteps=10_000)

# 6. After training, run a prediction
# 1. Reset environment
obs, info = env.reset()
done = False
total_reward = 0

while not done:
    # 2. Get valid action mask manually (because env is wrapped)
    action_mask = env.action_masks()
    print("Action mask:", action_mask)
    # 3. Predict action with action mask
    action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
    
    # 4. Take a step in the environment
    obs, reward, done, truncated, info = env.step(action)
    
    # 5. Accumulate reward
    total_reward += reward

    # Optional: print what is happening
    print(f"Action taken (flat): {action} | Reward: {reward} | Done: {done}")

# 6. Final result
print(f"Episode finished. Total reward: {total_reward}")


