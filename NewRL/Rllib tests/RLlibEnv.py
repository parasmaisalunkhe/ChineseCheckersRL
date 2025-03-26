from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import Discrete, Box, Dict, MultiDiscrete, Sequence
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
def ChineseCheckersPattern():
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

class ChineseCheckers(MultiAgentEnv):

    def __init__(self, config=None):
        super().__init__()
        self.n_players = NUM_PLAYERS
        self.agents = self.possible_agents = ["agent_" + str(r) for r in range(NUM_PLAYERS)]
        self.observation_space = dict()
        self.action_space = dict()
        self.last_move = None
        self.num_moves = 0
        for agnt in self.agents:
            self.action_space[agnt] = MultiDiscrete([WIDTH*HEIGHT, WIDTH*HEIGHT])
            self.observation_space[agnt] = Dict({
                # "observation": Box(low=0, high=1, shape=(4 * self.n + 1, 4 * self.n + 1, 8)),
                "observation": Box(low=0, high=2, shape=(HEIGHT*WIDTH,)),
                "action_mask": Sequence(MultiDiscrete([HEIGHT*WIDTH, HEIGHT*WIDTH]))
            })
        self.global_board = ChineseCheckersPattern()
        self.startingIndexes = [i for i, char in enumerate(self.global_board) if char == 1]
        if self.n_players == 2:
            StartingListInitialize = [StartingPositions[0], StartingPositions[3]]
        elif self.n_players == 3:
            StartingListInitialize = [StartingPositions[0], StartingPositions[2], StartingPositions[4]]
        elif self.n_players == 4:
            StartingListInitialize = [StartingPositions[0], StartingPositions[1], StartingPositions[3], StartingPositions[4]]
        elif self.n_players == 6:
            StartingListInitialize = StartingPositions
        # print(self.global_board)
        for i,x in enumerate(StartingListInitialize):
                for pos in x:
                    self.global_board[self.startingIndexes[pos]] = i+2
        
    def reset(self, *, seed=None, options=None):
        self.num_moves = 0
        agentObs = dict()
        for agnt in self.agents:
            agentObs[agnt] = 0
        return agentObs, {}

    def step(self, action_dict):
        # return observation dict, rewards dict, termination/truncation dicts, and infos dict
        return {"agent_1": [None]}, {...}, ...

env = ChineseCheckers()
for i in range(HEIGHT):
    row = " ".join(str(x) for x in env.global_board[i*WIDTH:(i+1)*WIDTH])
    print(row)

# print(env.global_board)