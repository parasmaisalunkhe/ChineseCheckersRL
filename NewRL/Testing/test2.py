from gymnasium.spaces import Sequence, MultiDiscrete
actionsapce = Sequence(MultiDiscrete([3,4]))
print(actionsapce.sample())