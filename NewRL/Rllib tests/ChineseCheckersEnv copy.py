# ChineseCheckersEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import Discrete, Box, Dict, MultiDiscrete, Sequence
import gymnasium
import numpy as np
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
        
class ChineseCheckersEnv(MultiAgentEnv):
    metadata = {'render.modes': ['human']}
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

    def __init__(self, verbose = False, manual = False):
        super(ChineseCheckersEnv, self).__init__()
        self.name = 'chinesecheckers'
        self.manual = manual
        self.gridShape = (height, width)
        self.n_players = 6
        self.board = self.ChineseCheckersPattern()
        self.action_space = dict()
        self.observation_space = dict()
        self.startingIndexes = [i for i, char in enumerate(self.board) if char.symbol == 'X']
        self.agents = self.possible_agents = ["agent_" + str(r) for r in range(self.n_players)]
        for agnt in self.agents:
            self.action_space[agnt] = MultiDiscrete([width*height, width*height])
            self.observation_space[agnt] = Dict({
                "observation": Box(low=0, high=2, shape=(width*height,)),
                "action_mask": Sequence(MultiDiscrete([width*height, width*height]))
            })
        self.players = [Player('0', Token('&', 0)), Player('1', Token('$', 1)), Player('2', Token('@', 2)), Player('3', Token('#', 2)), Player('4', Token('%', 4)), Player('5', Token('*', 5))]
        self.PlrClr = {'&':'red', '$':"yellow", "@":"blue", "#":"green", "%": "magenta", "*":"cyan"}
        if self.n_players == 2:
            StartingListInitialize = [StartingPositions[0], StartingPositions[3]]
            self.rotates = [0,3]
        elif self.n_players == 3:
            StartingListInitialize = [StartingPositions[0], StartingPositions[2], StartingPositions[4]]
            self.rotates = [0,-2,2]
        elif self.n_players == 4:
            StartingListInitialize = [StartingPositions[0], StartingPositions[1], StartingPositions[3], StartingPositions[4]]
            self.rotates = [0,-1,3,2]
        elif self.n_players == 6:
            StartingListInitialize = StartingPositions
            self.rotates = [0,-1,-2,3,2,1]
        for i,x in enumerate(StartingListInitialize):
                for pos in x:
                    self.board[self.startingIndexes[pos]] = Token(self.players[i].token.symbol, self.players[i].token.id)
        self.manual = manual
        self.current_player_num = 0
        self.num_squares = width * height
        self.verbose = verbose
    
    @property
    def observation(self):
        return np.array([x.id for x in self.board]).reshape(self.gridShape)
    @property
    def current_player(self):
        return self.players[self.current_player_num]
    def boardPOVPlayer(self, player):
        rotationMove = self.rotates[player]
        if rotationMove < 0:
            board = self.rotate(self.board, dir="CCW")
        if rotationMove > 0:
            board = self.rotate(self.board, dir="CW")
        return board
    
    def rotate(self, board, dir="CW"):
        template = [10,23,11,35,24,12,46,36,25,13,98,86,75,65,56,47,37,26,14,6,3,1,0,99,87,76,66,57,48,38,27,15,7,4,2,100,88,77,67,58,49,39,28,16,8,5,101,89,78,68,59,50,40,29,17,9,102,90,79,69,60,51,41,30,18,111,103,91,80,70,61,52,42,31,19,115,112,104,92,81,71,62,53,43,32,20,118,116,113,105,93,82,72,63,54,44,33,21,120,119,117,114,106,94,83,73,64,55,45,34,22,107,95,84,74,108,96,85,109,97,110]
        board = [Token('.', -2)]*(width*height)
        for i,x in enumerate(template):
            if dir == "CCW":
                board[self.startingIndexes[x]] = self.board[self.startingIndexes[i]]
            elif dir == "CW":
                board[self.startingIndexes[i]] = self.board[self.startingIndexes[x]]
        return board
    

    def all_legal_actions(self, board):
        legal_actions = []
        for index,x in enumerate(board):
            if x.id == self.current_player_num:
                AllValidmoves = self.TheListofAllPossibleMoves(index, board)
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
                if mv[0] > 0 and mv[0] < width*height and mv[1] > 0 and mv[1] < width*height and board[mv[0]].id != -1 and board[mv[0]].id != -2 and board[mv[1]].id == -1:
                    if mv[1] not in newCallStack:
                        possibleFurtherJumps.add(mv[1])
                        newCallStack.add(mv[1])
            return list(LegalMoves) + list(self.jumpHelper(possibleFurtherJumps, newCallStack, board))
        return list(LegalMoves)
            
    def TheListofAllPossibleMoves(self, index, board):
        possibleSteps = set()
        possibleJumps = set()
        posOneStepMoves = [(index+2, index+4), (index-2, index-4), (index-width+1,index-2*width+2), (index-width-1, index-2*width-2), (index+width+1, index+2*width+2), (index+width-1, index+2*width-2)]
        for x in posOneStepMoves:
            if x[0] > 0 and x[0] < width*height:
                if board[x[0]].id == -1:
                    possibleSteps.add(x[0])
                else:
                    if x[1] > 0 and x[1] < width*height and board[x[1]].id == -1:
                        possibleJumps.add(x[1])
        if not possibleJumps:
            return list(possibleSteps)
        return list(possibleSteps) + list(self.jumpHelper(possibleJumps, set(), board))
        
    def isLegal(self, action):
        return np.any(self.legal_actions == action)
    
    def step(self, action, validmoves):
        reward = 0
        boardPOV = self.boardPOVPlayer(self.current_player_num)
        if action in validmoves:
            boardPOV[action[1]].symbol = self.players[self.current_player_num].token.symbol
            boardPOV[action[1]].id = self.current_player_num
            boardPOV[action[0]].symbol = "X"
            boardPOV[action[0]].id = -1
            self.turns_taken += 1
            reward, done = self.check_game_over()
        self.done = done
        if not done:
            self.current_player_num += 1
            if self.current_player_num == self.n_players:
                self.current_player_num = 0
        obs = self.observation(self, boardPOV, self.current_player_num)
        return {obs, reward, done, {}, {}}
    
    def observation(self, board, number):
        listBoard = []
        for x in board:
            if x.symbol == "X" or x.symbol == ".":
                listBoard.append(x.id + 1)
            else:
                if x.id == number:
                    listBoard.append(2)
                else:
                    listBoard.append(1)
        return np.array(listBoard)
    
    def reset(self, seed=None):
        self.board = self.ChineseCheckersPattern()
        self.current_player_num = 0
        self.turns_taken = 0
        self.done = False
        self.players = [Player('0', Token('&', 0)), Player('1', Token('$', 1)), Player('2', Token('@', 2)), Player('3', Token('#', 2)), Player('4', Token('%', 4)), Player('5', Token('*', 5))]
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
        
        return self.observation(self.board, 0), {}

    def check_game_over(self):
        for i,x in enumerate(self.board):
            if x.id == self.current_player_num:
                if i not in WinningPositions[x.id]:
                    return -0.1, False
        return 10, True
    
    def render(self, board, mode='human', close=False):
        for i in range(height):
            row = " ".join(str(x.symbol) if x.symbol == '.' or x.symbol == "X" else colored(str(x.symbol), self.PlrClr[x.symbol]) for x in board[i*width:(i+1)*width])
            print(row)

env = ChineseCheckersEnv()
env.render(env.board)
print("Rotated Clockwise Board")
env.render(env.rotate(env.board))
print("Rotated CCW Board")
env.render(env.rotate(env.board, dir="CCW"))