import numpy as np
import gymnasium
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium.spaces import Discrete, Box, Dict, MultiDiscrete, Sequence
from termcolor import colored
width = 29
height = 19
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
class Player():
    def __init__(self, id, tokens):
        self.id = id
        self.token = tokens
        
class Token():
    def __init__(self, symbol, id):
        self.id = id
        self.symbol = symbol


class ChineseCheckersEnvAEC(AECEnv):
    metadata = {"render_modes": ["human"], "name": "ChineseCheckersAEC_v0"}
    def ChineseCheckersPattern(self):
        finalpattern = "." * width
        holes = [1,2,3,4,13,12,11,10,9,10,11,12,13,4,3,2,1]  # holes = [1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1]
        for n in holes:
            pattern = ""
            for i in range(n):
                pattern += "X."
            pattern = pattern[:-1]
            while len(pattern) != width:
                pattern = "." + pattern + "."
            finalpattern += pattern
        finalpattern += "." * width
        return [Token('X', -1) if char == 'X' else Token('.', -2) for char in finalpattern]

    def __init__(self, render_mode = None, n_players = 2, max_steps = 10000):
        self.current_step = 0
        self.gridShape = (height, width)
        self.max_steps = max_steps
        self.n_players = n_players
        self.board = self.ChineseCheckersPattern()
        self.possible_agents = [f"player_{r}" for r in range(self.n_players)]
        self.agents = [f"player_{r}" for r in range(self.n_players)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.startingIndexes = [i for i, char in enumerate(self.board) if char.symbol == 'X']
        self.players = [Player('0', Token('&', 0)), Player('1', Token('$', 1)), Player('2', Token('@', 2)), Player('3', Token('#', 2)), Player('4', Token('%', 4)), Player('5', Token('*', 5))]
        self.PlrClr = {'&':'red', '$':"yellow", "@":"blue", "#":"green", "%": "magenta", "*":"cyan"}
        if self.n_players == 2:
            StartingListInitialize = [StartingPositions[0], StartingPositions[3]]
        elif self.n_players == 3:
            StartingListInitialize = [StartingPositions[0], StartingPositions[2], StartingPositions[4]]
        elif self.n_players == 4:
            StartingListInitialize = [StartingPositions[0], StartingPositions[1], StartingPositions[3], StartingPositions[4]]
        elif self.n_players == 6:
            StartingListInitialize = StartingPositions
        for i,x in enumerate(StartingListInitialize):
                for pos in x:
                    self.board[self.startingIndexes[pos]] = Token(self.players[i].token.symbol, self.players[i].token.id)
        self.num_squares = width * height
        self.action_space = {agent : gymnasium.spaces.MultiDiscrete([height*width, height*width]) for agent in self.agents}
        # self.observation_space = {agent : gymnasium.spaces.Box(-2, self.n_players-1, (self.gridShape), dtype=np.int64) for agent in self.agents}
        self.observation_space = {
            agent: Dict({
                "observation" : Box(low=0, high=1, shape=(120,), dtype=np.int64),
                "action_mask" : Sequence(MultiDiscrete(low=0, height=height*width, shape=(height*width, ), dytpe=np.int64))
            })
            for agent in self.agents
        }

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
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
    def all_legal_actions(agent, self):
        legal_actions = []
        for index,x in enumerate(self.board):
            if x.id == self.agent_name_mapping[agent]:
                AllValidmoves = self.TheListofAllPossibleMoves(index)
                tuples = [(index, num) for num in AllValidmoves]
                legal_actions += tuples
        return legal_actions

    def jumpHelper(self, JumpsLegal, callStack):
        newCallStack = callStack
        LegalMoves = JumpsLegal
        for index in LegalMoves:
            possibleFurtherJumps = set()
            posJumpMoves = [(index+2, index+4), (index-2, index-4), (index-width+1,index-2*width+2), (index-width-1, index-2*width-2), (index+width+1, index+2*width+2), (index+width-1, index+2*width-2)]
            for mv in posJumpMoves:
                if mv[0] > 0 and mv[0] < width*height and mv[1] > 0 and mv[1] < width*height and self.board[mv[0]].id != -1 and self.board[mv[0]].id != -2 and self.board[mv[1]].id == -1:
                    if mv[1] not in newCallStack:
                        possibleFurtherJumps.add(mv[1])
                        newCallStack.add(mv[1])
            return list(LegalMoves) + list(self.jumpHelper(possibleFurtherJumps, newCallStack))
        return list(LegalMoves)
            
    def TheListofAllPossibleMoves(self, index):
        possibleSteps = set()
        possibleJumps = set()
        posOneStepMoves = [(index+2, index+4), (index-2, index-4), (index-width+1,index-2*width+2), (index-width-1, index-2*width-2), (index+width+1, index+2*width+2), (index+width-1, index+2*width-2)]
        for x in posOneStepMoves:
            if x[0] > 0 and x[0] < width*height:
                if self.board[x[0]].id == -1:
                    possibleSteps.add(x[0])
                else:
                    if x[1] > 0 and x[1] < width*height and self.board[x[1]].id == -1:
                        possibleJumps.add(x[1])
        if not possibleJumps:
            return list(possibleSteps)
        return list(possibleSteps) + list(self.jumpHelper(possibleJumps, set()))
    def isLegal(self, action):
        return np.any(self.all_legal_actions() == action)
    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        self.state[self.agent_selection]
        

        if not self.isLegal(action): 
            done = True
            reward = -1
        else:
            self.board[action[1]].symbol = self.players[self.current_player_num].token.symbol
            self.board[action[1]].id = self.current_player_num
            self.board[action[0]].symbol = "X"
            self.board[action[0]].id = -1
            self.turns_taken += 1
            reward, done = self.check_game_over()
        self.done = done

        if not done:
            self.current_player_num += 1
            if self.current_player_num == self.n_players:
                self.current_player_num = 0

        # return np.array([x.id for x in self.board]).reshape(self.gridShape), reward, done, False, {}
        if self.render_mode == "human":
            self.render()
    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        for i in range(height):
            row = " ".join(str(x.symbol) if x.symbol == '.' or x.symbol == "X" else colored(str(x.symbol), self.PlrClr[x.symbol]) for x in self.board[i*width:(i+1)*width])
            print(row)

    def close(self):
        pass

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        return np.array(self.observations[agent])
    def observation_space(self, agent):
        return gymnasium.spaces.MultiDiscrete([height*width, height*width])
    def action_space(self, agent):
        return gymnasium.spaces.Box(-2, self.n_players-1, (self.gridShape), dtype=np.int64)
def env():
    return wrappers.CaptureStdoutWrapper(ChineseCheckersEnvAEC())

