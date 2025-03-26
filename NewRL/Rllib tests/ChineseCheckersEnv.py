# ChineseCheckersEnv
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
        
class ChineseCheckersEnv(gymnasium.Env):
    metadata = {'render.modes': ['human']}
    def ChineseCheckersPattern(self):
        finalpattern = "." * width
        # finalpattern += "#" * width
        holes = [1,2,3,4,13,12,11,10,9,10,11,12,13,4,3,2,1]  # holes = [1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1]
        for n in holes:
            pattern = ""
            for i in range(n):
                pattern += "X."
            pattern = pattern[:-1]
            while len(pattern) != width:
                pattern = "." + pattern + "."
            # print(pattern)
            finalpattern += pattern
        finalpattern += "." * width
        return [Token('X', -1) if char == 'X' else Token('.', -2) for char in finalpattern]

    def __init__(self, verbose = False, manual = False):
        super(ChineseCheckersEnv, self).__init__()
        self.name = 'chinesecheckers'
        self.manual = manual
        self.gridShape = (height, width)
        self.n_players = 2
        self.board = self.ChineseCheckersPattern()
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
        self.manual = manual
        self.current_player_num = 0
        self.num_squares = width * height
        self.action_space = gymnasium.spaces.MultiDiscrete([height*width, height*width])
        self.observation_space = gymnasium.spaces.Box(-2, self.n_players-1, (self.gridShape), dtype=np.int64)
        self.verbose = verbose
    

    @property
    def observation(self):
        return np.array([x.id for x in self.board]).reshape(self.gridShape)
    @property
    def current_player(self):
        return self.players[self.current_player_num]
    
    @property
    def legal_actions(self):
        return self.all_legal_actions()

    def all_legal_actions(self):
        legal_actions = []
        for index,x in enumerate(self.board):
            if x.id == self.current_player_num:
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
        return np.any(self.legal_actions == action)
    
    def step(self, action):

        reward = 0

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

        return np.array([x.id for x in self.board]).reshape(self.gridShape), reward, done, False, {}

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
        return self.observation, {}

    def check_game_over(self):
        for i,x in enumerate(self.board):
            if x.id == self.current_player_num:
                if i not in WinningPositions[x.id]:
                    return -0.1, False
        return 10, True
    
    def render(self, mode='human', close=False):
        for i in range(height):
            row = " ".join(str(x.symbol) if x.symbol == '.' or x.symbol == "X" else colored(str(x.symbol), self.PlrClr[x.symbol]) for x in self.board[i*width:(i+1)*width])
            print(row)

