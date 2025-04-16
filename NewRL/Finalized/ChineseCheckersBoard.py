import numpy as np
from termcolor import colored
import gymnasium
from gymnasium.spaces import MultiDiscrete, Box, Dict, Sequence
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from ray.rllib.env.multi_agent_env import MultiAgentEnv
width = 29
height = 19

def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = ChineseCheckersBoard(6)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class ChineseCheckersBoard(AECEnv):
    metadata = {"render_modes": ["ansi"], "name": "ChineseCheckers"}
    def __init__(self, numPlayers, render_mode=None):
        self.StartingPositions = [
            [0,1,2,3,4,5,6,7,8,9],
            [19,20,21,22,32,33,34,44,45,55],
            [74,84,85,95,96,97,107,108,109,110],
            [111,112,113,114,115,116,117,118,119,120],
            [65,75,76,86,87,88,98,99,100,101],
            [10,11,12,13,23,24,25,35,36,46],
        ]
        self.WinningPositions = [
            [111,112,113,114,115,116,117,118,119,120],
            [65,75,76,86,87,88,98,99,100,101],
            [10,11,12,13,23,24,25,35,36,46],
            [0,1,2,3,4,5,6,7,8,9],
            [19,20,21,22,32,33,34,44,45,55],
            [74,84,85,95,96,97,107,108,109,110],
        ]
        self.StartingLocations = {2: [0,3], 3: [0,2,4], 4: [0,1,3,4], 6: [0,1,2,3,4,5]}.get(numPlayers)
        self.EndingLocations = {2: [3,0], 3: [3,5,1], 4: [3,4,1,2], 6: [3,4,5,0,1,2]}.get(numPlayers)
        self.PlayerPOVPosition = {2: [3], 3: [2], 4: [1,2], 6: [1]}.get(numPlayers) #CCW
        self.emptyTokenLocations = None
        self.GlobalBoard = self.ChineseCheckersPattern()
        self.ActualEndingLocations = [self.emptyTokenLocations[x] for x in self.WinningPositions[1]]


        self.possible_agents = ["player_" + str(r) for r in range(numPlayers)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self._action_spaces = {agent: MultiDiscrete([width*height, width*height]) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: Dict({
                "board": Box(low=-1, high=2, shape=(width * height,), dtype=np.int32),
                "action_mask": Sequence(MultiDiscrete([width * height, width * height]))
            }) for agent in self.possible_agents
        }
        self.render_mode = render_mode

    def observation_space(self, agent):
        return MultiDiscrete([width*height, width*height])
    def action_space(self, agent):
        return Dict({
                "board": Box(low=-1, high=2, shape=(width * height,), dtype=np.int32),
                "action_mask": Sequence(MultiDiscrete([width * height, width * height]))
            })
    def ChineseCheckersPattern(self):
        Dict = {"X": 0, ".": -1}
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
        newBoard = np.array([Dict[char] for char in finalpattern])
        self.emptyTokenLocations = list(np.where(newBoard == 0)[0])
        for i,loc in enumerate(self.StartingLocations):
            for x in self.StartingPositions[loc]:
                newBoard[self.emptyTokenLocations[x]] = i+1
        return newBoard

    def render(self, board):
        """Prints ASCII representations of the Global board."""
        PlayertoColor = ["black", "white", "yellow", "blue", "green", "magenta", "cyan", "red"]
        for i in range(height):
            row = " ".join(colored(str(x) if x != -1 else " ", PlayertoColor[x+1]) for x in board[i*width:(i+1)*width])
            print(row)

    def nextPlayerPOV(self):
        board = None
        rotationMove = self.PlayerPOVPosition.pop(0)
        board = self.rotateNtimes(self.GlobalBoard, rotationMove)
        self.PlayerPOVPosition.append(rotationMove)
        return board

    def rotate(self, board):
        newboard = board.copy()
        template = [10,23,11,35,24,12,46,36,25,13,98,86,75,65,56,47,37,26,14,6,3,1,0,99,87,76,66,57,48,38,27,15,7,4,2,100,88,77,67,58,49,39,28,16,8,5,101,89,78,68,59,50,40,29,17,9,102,90,79,69,60,51,41,30,18,111,103,91,80,70,61,52,42,31,19,115,112,104,92,81,71,62,53,43,32,20,118,116,113,105,93,82,72,63,54,44,33,21,120,119,117,114,106,94,83,73,64,55,45,34,22,107,95,84,74,108,96,85,109,97,110]
        for i,x in enumerate(template):
            newboard[self.emptyTokenLocations[x]] = board[self.emptyTokenLocations[i]]
        return newboard
    
    def rotateNtimes(self, board, n):
        for i in range(n):
            board = self.rotate(board)
        return board
    
    def allLegalActions(self, board, player_num):
        legal_actions = []
        for index,x in enumerate(board):
            if x == player_num:
                inEndzone = index in self.ActualEndingLocations #board rotation
                AllValidmoves = self.TheListofAllPossibleMoves(index, board, inEndzone)
                tuples = [(index, num) for num in AllValidmoves]
                legal_actions += tuples
        return legal_actions

    def jumpHelper(self, JumpsLegal, callStack, board):
        newCallStack = callStack
        LegalMoves = JumpsLegal
        for index in LegalMoves:
            possibleFurtherJumps = set()
            posJumpMoves = [(index+2, index+4), (index-2, index-4), (index-width+1,index-2*width+2), (index-width-1, index-2*width-2), (index+width+1, index+2*width+2), (index+width-1, index+2*width-2)]
            for mv in posJumpMoves:
                if mv[0] > 0 and mv[0] < width*height and mv[1] > 0 and mv[1] < width*height and board[mv[0]] != 0 and board[mv[0]] != -1 and board[mv[1]] == 0:
                    if mv[1] not in newCallStack:
                        possibleFurtherJumps.add(mv[1])
                        newCallStack.add(mv[1])
            return list(LegalMoves) + list(self.jumpHelper(possibleFurtherJumps, newCallStack, board))
        return list(LegalMoves)
            
    def TheListofAllPossibleMoves(self, index, board, Endzone=False):
        possibleSteps = set()
        possibleJumps = set()
        posOneStepMoves = [(index+2, index+4), (index-2, index-4), (index-width+1,index-2*width+2), (index-width-1, index-2*width-2), (index+width+1, index+2*width+2), (index+width-1, index+2*width-2)]
        for x in posOneStepMoves:
            if x[0] > 0 and x[0] < width*height:
                if board[x[0]] == 0:
                    if not Endzone or (Endzone and x[0] in self.ActualEndingLocations):
                        possibleSteps.add(x[0])
                else:
                    if x[1] > 0 and x[1] < width*height and board[x[1]] == 0 and board[x[1]] != -1:
                        if not Endzone or (Endzone and x[1] in self.ActualEndingLocations):
                            possibleJumps.add(x[1])
        if not possibleJumps:
            return list(possibleSteps)
        return list(possibleSteps) + list(self.jumpHelper(possibleJumps, set(), board))
        
    def isLegal(self, action, board, player_num):
        return np.any(self.allLegalActions(board, player_num) == action)
    
    def isGameOver(self, board, player_num):
        endLocation = [board[x] for x in self.ActualEndingLocations]

        if len(set(endLocation)) == 1 and list(set(endLocation))[0] == player_num:
            return True
        else:
            if 0 in endLocation:
                return False
            else:
                return True
            
    def step(self, board, action, player_num):
        board[action[0]] = 0
        board[action[1]] = player_num
        return np.where((board == -1) | (board == 0), board, np.where(board == player_num, 2, 1)), -1, False, board

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

# env = ChineseCheckersBoard(6)
# env.render(env.GlobalBoard)
# env.render(env.rotate(env.GlobalBoard))
# env.render(env.rotateNtimes(env.GlobalBoard, 3))
