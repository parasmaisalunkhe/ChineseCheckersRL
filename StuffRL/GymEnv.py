import numpy as np
from termcolor import colored
from gymnasium import spaces
import gymnasium as gym
width = 29
height = 19

class ChineseCheckersBoard(gym.Env):
    def __init__(self, n_players):
        super().__init__()
        self.numPlayers = n_players
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
        
        self.StartingLocations = {2: [0,3], 3: [0,2,4], 4: [0,1,3,4], 6: [0,1,2,3,4,5]}.get(self.numPlayers)
        self.EndingLocations = {2: [3,0], 3: [3,5,1], 4: [3,4,1,2], 6: [3,4,5,0,1,2]}.get(self.numPlayers)
        self.PlayerPOVPosition = {2: [3,3], 3: [2,2,3], 4: [1,2,1,2], 6: [1,1,1,1,1,1]}.get(self.numPlayers) #CCW
        self.emptyTokenLocations = None
        
        self.GlobalBoard = self.ChineseCheckersPattern().astype(np.int32)
        self.currentPlayerBoardView = self.GlobalBoard.copy()
        self.ActualEndingLocations = [self.emptyTokenLocations[x] for x in self.WinningPositions[1]]

        self.agents = self.possible_agents = ["player_" + str(r) for r in range(1,self.numPlayers+1)]
        self.agentsID = {item: idx + 1 for idx, item in enumerate(self.agents)}
        self.IDagents = {idx + 1: item for idx, item in enumerate(self.agents)}

        self.current_player = None 

        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=-1, high=2, shape=(width * height,), dtype=np.float64),
            "action_mask": spaces.Sequence(spaces.MultiDiscrete([width*height, width*height]))
        })
        self.action_space = spaces.MultiDiscrete([width*height, width*height])

        self.last_move = None
        self.num_moves = 0
    
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
        return newBoard.astype(dtype=np.float32)

    def render(self, board):
        """Prints ASCII representations of the Global board."""
        board = board.astype(np.int32)
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
                AllValidmoves = self.TheListofAllPossibleMoves(index, board)
                tuples = [np.array([index, num]) for num in AllValidmoves]
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
            
    def TheListofAllPossibleMoves(self, index, board):
        board = board.astype(np.int32)
        possibleSteps = set()
        possibleJumps = set()
        # print(index)
        posOneStepMoves = [(index+2, index+4), (index-2, index-4), (index-width+1,index-2*width+2), (index-width-1, index-2*width-2), (index+width+1, index+2*width+2), (index+width-1, index+2*width-2)]
        for x in posOneStepMoves:
            if x[0] > 0 and x[0] < width*height:
                # print(board[index], board[x[0]])
                if board[x[0]] == 0:
                    # if index in self.ActualEndingLocations and x[0] in self.ActualEndingLocations:
                    possibleSteps.add(x[0])
                    # if index not in self.ActualEndingLocations:
                    #     possibleSteps.add(x[0])
                elif x[1] > 0 and x[1] < width*height and board[x[1]] == 0:
                    # print(board[index], board[x[1]])
                    # if index in self.ActualEndingLocations and x[1] in self.ActualEndingLocations:
                    possibleSteps.add(x[1])
                    # if index not in self.ActualEndingLocations:
                        # possibleSteps.add(x[1])
        if not possibleJumps:
            return list(possibleSteps)
        return list(possibleSteps) + list(self.jumpHelper(possibleJumps, set(), board))
        
    def isLegal(self, action, board, player_num):
        lists = self.allLegalActions(board, player_num)
        for x in lists:
            if np.array_equal(x, action):
                return True
        return False
    
    def isGameOver(self, board, player_num):
        endLocation = [board[x] for x in self.ActualEndingLocations]
        if len(set(endLocation)) == 1 and list(set(endLocation))[0] == player_num:
            return True
        else:
            if 0 in endLocation:
                return False
            else:
                return True
            
    def step(self, action):
        done = False
        action = np.array(action)
        # print("action", action)
        self.num_moves += 1
        
        rewards = -1.0
        # print("Key", self.current_player)
        board = self.GlobalBoard[:]
        # print(self.IDagents[self.current_player])
        if not self.isLegal(action, self.GlobalBoard, self.current_player):
            # print("wat")
            reward = -5.0
            done = True
            boardObservation = self.getObservation(board)
            validMoves = self.allLegalActions(self.GlobalBoard, self.current_player)
            # nextActionMask = self.generate_action_mask(validMoves)
            return {"obs": boardObservation, "action_mask": validMoves}, rewards, done, False, {}
        else:
            terminateds = False
            Token = self.current_player
            self.num_moves += 1
            board[action[0]] = 0
            board[action[1]] = Token
            if self.isGameOver(board, Token):
                self.rewards = 10.0
                terminateds = True
            # rotationMove = self.PlayerPOVPosition.pop(0)
            self.GlobalBoard = self.nextPlayerPOV()
            self.next_player()
            boardObservation = self.getObservation(board)
            validMoves = self.allLegalActions(self.GlobalBoard, self.current_player)
            # nextActionMask = self.generate_action_mask(validMoves)
            return {"obs": boardObservation, "action_mask": validMoves}, rewards, done, False, {}
    
    def reset(self, seed=None):
        self.num_moves = 0
        self.current_player = 1
        self.GlobalBoard = self.ChineseCheckersPattern()
        validMoves = self.allLegalActions(self.GlobalBoard, self.current_player)
        # actionMask = self.generate_action_mask(validMoves)
        board = self.GlobalBoard
        self.num_moves = 0
        # obs = np.where((board == -1) | (board == 0), board, np.where(board == self.current_player, 2, 1)).astype(np.float32)
        boardObs = self.getObservation(board)
        observation = {"obs": boardObs, "action_mask": validMoves}
        return observation, {}
    
    def next_player(self):
        self.current_player += 1
        if self.current_player == self.numPlayers + 1:
            self.current_player = 1

    def getObservation(self, board):
        return np.where((board == -1) | (board == 0), board, np.where(board == self.current_player, 2, 1)).astype(np.float32)
    # def generate_action_mask(self):
    #     # print(valid_moves)
    #     action_mask = np.full((width*height, 2), False)

    #     return action_mask
    
# from stable_baselines3.common.env_checker import check_env
# check_env(ChineseCheckersBoard(6))