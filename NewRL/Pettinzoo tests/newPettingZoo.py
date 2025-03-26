import gymnasium
import numpy as np
from pettingzoo import AECEnv
import functools
from pettingzoo.utils import agent_selector, wrappers
from gymnasium.spaces import Discrete, Box, Dict, MultiDiscrete, Sequence

#-------------------------------------------------------------
WIDTH = 29
HEIGHT = 19
NUM_PLAYERS = 2
StartingPositions = [
    [0,1,2,3,4,5,6,7,8,9],
    [19,20,21,22,32,33,34,44,45,55],
    [74,84,85,95,96,97,107,108,109,110],
    [111,112,113,114,115,116,117,118,119,120],
    [65,75,76,86,87,88,98,99,100,101],
    [10,11,12,13,23,24,25,35,36,46],
]
WinningPositions = [
    [111,112,113,114,115,116,117,118,119,120],
    [65,75,76,86,87,88,98,99,100,101],
    [10,11,12,13,23,24,25,35,36,46],
    [0,1,2,3,4,5,6,7,8,9],
    [19,20,21,22,32,33,34,44,45,55],
    [74,84,85,95,96,97,107,108,109,110],
]
def ChineseCheckersPattern(self):
        finalpattern = "." * WIDTH
        # finalpattern += "#" * width
        holes = [1,2,3,4,13,12,11,10,9,10,11,12,13,4,3,2,1]  # holes = [1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1]
        for n in holes:
            pattern = ""
            for i in range(n):
                pattern += "X."
            pattern = pattern[:-1]
            while len(pattern) != WIDTH:
                pattern = "." + pattern + "."
            # print(pattern)
            finalpattern += pattern
        finalpattern += "." * WIDTH
        return [1 if char == 'X' else 0 for char in finalpattern]
GLOBAL_BOARD = ChineseCheckersPattern()

#-------------------------------------------------------------

class ChineseCheckers(AECEnv):
    metadata = {"render_modes": ["human", "ascii"], "name": "ChineseCheckers_v2"}
    
    def __init__(self, render_mode=None):

        self.possible_agents = ["player_" + str(r) for r in range(NUM_PLAYERS)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self._action_spaces = {agent: MultiDiscrete(WIDTH*HEIGHT, WIDTH*HEIGHT) for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.state = GLOBAL_BOARD
        self.observation_spaces = {
            agent: Dict({
                # "observation": Box(low=0, high=1, shape=(4 * self.n + 1, 4 * self.n + 1, 8)),
                "observation": Box(low=0, high=2, shape=(HEIGHT*WIDTH,)),
                "action_mask": Sequence(MultiDiscrete([HEIGHT*WIDTH, HEIGHT*WIDTH]))
            })
            for agent in self.agents
        }
        self.render_mode = render_mode

    
    def observation_space(self, agent):
         return super().observation_space(agent)
    
    def action_space(self, agent):
         return super().action_space(agent)
    
    def render(self):
         ...
    def observe(self, agent):
         ...
    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = GLOBAL_BOARD
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
    def step(self, action):
         ...
    def reset(self):
         


