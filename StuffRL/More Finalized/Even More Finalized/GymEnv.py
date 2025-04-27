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
        
        self.StartingLocations = {2: [0,3], 3: [0,3,3], 4: [0,1,3,4], 6: [0,1,2,3,4,5]}.get(self.numPlayers)
        self.PlayerPOVPosition = {2: [3,3], 3: [2,2,2], 4: [1,2,1,2], 6: [1,1,1,1,1,1]}.get(self.numPlayers) #CCW
        self.emptyTokenLocations = None
        
        self.GlobalBoard = self.ChineseCheckersPattern().astype(np.int32)
        self.currentPlayerBoardView = self.GlobalBoard.copy()
        self.ActualEndingLocations = [self.emptyTokenLocations[x] for x in self.WinningPositions[0]]

        self.agents = self.possible_agents = ["player_" + str(r) for r in range(1,self.numPlayers+1)]
        self.agentsID = {item: idx + 1 for idx, item in enumerate(self.agents)}
        self.IDagents = {idx + 1: item for idx, item in enumerate(self.agents)}

        self.current_player = None 

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(100,), dtype=np.float64)
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
        PlayertoColor = ["black", "white", "yellow", "blue", "greennnn", "magenta", "cyan", "red"]
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
    
    @property
    def allLegalActions(self):
        board = self.GlobalBoard
        player_num = self.current_player
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
        posOneStepMoves = [(index+2, index+4), (index-2, index-4), (index-width+1,index-2*width+2), (index-width-1, index-2*width-2), (index+width+1, index+2*width+2), (index+width-1, index+2*width-2)]
        for x in posOneStepMoves:
            if index in self.ActualEndingLocations:
                if x[0] in self.ActualEndingLocations:
                    possibleSteps.add(x[0])
                if x[1] in self.ActualEndingLocations:
                    possibleSteps.add(x[0])
            elif x[0] > 0 and x[0] < width*height:
                if board[x[0]] == 0:
                    possibleSteps.add(x[0])
                elif x[1] > 0 and x[1] < width*height and board[x[1]] == 0:
                    possibleSteps.add(x[1])
        if not possibleJumps:
            return list(possibleSteps)
        return list(possibleSteps) + list(self.jumpHelper(possibleJumps, set(), board))
        
    def valid_action_mask(self):
        """Return a flattened action mask for the flattened action space."""
        mask = np.zeros((width*height, width*height), dtype=bool)
        for x in self.allLegalActions:
            mask[x[0], x[1]] = True  # (2,1)
        newMask = mask.flatten()
        print(len(newMask))
        return newMask
    
    def isGameOver(self, board, player_num):
        if self.num_moves <= 5:
            return False
        endLocation = [board[x] for x in self.ActualEndingLocations]
        if len(set(endLocation)) == 1 and list(set(endLocation))[0] == player_num:
            return True
        else:
            if 0 in endLocation:
                return False
            else:
                return True
            
    def step(self, action):
        action = np.array(action)
        # if self.valid_action_mask()[action[0] * width*height + action[1]] == False:
        #     return self.getObservation(), -2, True, True, {}
        self.num_moves += 1
        reward = 0
        board = self.GlobalBoard[:]
        Token = self.current_player
        board[action[0]] = 0
        board[action[1]] = Token
        done = self.isGameOver(board, Token) or self.num_moves >= 200
        if action[1] in self.ActualEndingLocations:
            reward = 5.0
        if done:
            reward = 10.0
        self.GlobalBoard = board
        observation = self.getObservation()
        self.GlobalBoard = self.nextPlayerPOV()
        self.next_player()
        return observation, reward, done, False, {}
    
    def reset(self, seed=None, options=None):
        self.num_moves = 0
        self.current_player = 1
        self.GlobalBoard = self.ChineseCheckersPattern()
        return self.getObservation(), {}
    
    def next_player(self):
        self.current_player += 1
        if self.current_player == self.numPlayers + 1:
            self.current_player = 1
        
    def getObservation(self):
        indices = np.where(self.GlobalBoard == self.current_player)[0]
        print(indices)
        indiciesCoordinates = [(x // width, x % width) for x in indices]
        print(len(indiciesCoordinates))
        destination = [(x // width, x % width) for x in self.ActualEndingLocations]
        A = np.array(indiciesCoordinates)
        B = np.array(destination)
        distances = np.linalg.norm(A[:, np.newaxis] - B, axis=2)
        return distances.flatten()
    
    def numPiecesEndZone(self, board):
        count = 0 
        for x in self.ActualEndingLocations:
            if board[x] != ".":
                count += 1
        return count
    
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()

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
    
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.env_checker import check_env


env = ChineseCheckersBoard(2)


# obs, info = env.reset()
# print(obs)
env = FlattenMultiDiscreteWrapper(env)  # Flatten actions
env = ActionMasker(env, mask_fn)         # Add action masking
# check_env(env, warn=True)
# # 5. Create and train the MaskablePPO model
model = MaskablePPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

# # 6. After training, run a prediction
# # 1. Reset environment
# obs, info = env.reset()
# done = False
# total_reward = 0

# while not done:
#     # 2. Get valid action mask manually (because env is wrapped)
#     action_mask = env.action_masks()
#     print("Action mask:", action_mask)
#     # 3. Predict action with action mask
#     action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
    
#     # 4. Take a step in the environment
#     obs, reward, done, truncated, info = env.step(action)
    
#     # 5. Accumulate reward
#     total_reward += reward

#     # Optional: print what is happening
#     print(f"Action taken (flat): {action} | Reward: {reward} | Done: {done}")

# # 6. Final result
# print(f"Episode finished. Total reward: {total_reward}")
