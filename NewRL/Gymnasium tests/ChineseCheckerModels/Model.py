# from gym import Env, spaces
from gymnasium.spaces import Discrete, Box, MultiDiscrete
from stable_baselines3.common.env_checker import check_env
from collections import deque
from stable_baselines3 import PPO
import numpy as np
import gymnasium
import random
import math
import cv2
import os

import numpy as np
from gymnasium.spaces import MultiDiscrete

class DynamicMultiDiscrete(MultiDiscrete):
    def __init__(self, env):
        self.env = env
        initial_nvec = [121,121]
        super().__init__(initial_nvec)

    def update(self):
        """
        Dynamically update the action space's nvec.
        """
        self.nvec = np.array([121,121], dtype=np.int64)

    def sample(self):
        """
        Sample a random action from the dynamically defined MultiDiscrete space.
        """
        self.update()
        x = random.choice([i for i, c in enumerate(self.env.boardState) if c == self.env.player])
        mygoal = self.env.goalzones[self.env.args.index(self.env.player)]
        temp = random.choice(self.env.neighbors[x])
        if x in self.env.goalzones[self.env.args.index(self.env.player)]:
            while (type(temp) != int and temp[0] not in mygoal and temp[1] not in mygoal) or (type(temp) == int and temp not in mygoal):
                temp = random.choice(self.env.neighbors[x])
            if type(temp) == int:
                return (x,temp)
            else:
                choice = random.randint(0, 1)
                if choice == 1 and self.env.boardState[temp[0]] != "." and temp[1] in mygoal:
                    return (x,temp[1])
                else:
                    return (x,temp[0])
        if type(temp) == int:
            return (x, temp)
        choice = random.randint(0, 1)
        if choice == 1 and self.env.boardState[temp[0]] != ".":
            return (x,temp[1])
        else:
            return (x,temp[0])

class ChineseCheckersEnv(gymnasium.Env):
    def __init__(self, arg):
        self.args = arg
        self.rotate = [] #just for 2 players}
        self.players = arg[:]
        self.player = arg[0]
        self.action_space = MultiDiscrete([121,121])#DynamicMultiDiscrete(self)
        self.numPlayers = len(arg)
        self.observation_space = Box(low=0, high=self.numPlayers, shape=(121,), dtype=np.int32)
        self.state = None #board represent
        self.imgStore = []
        self.locationspix = []
        self.allzones = [[0,1,2,3,4,5,6,7,8,9], [19,20,21,22,32,33,34,44,45,55], [74,84,85,95,96,97,107,108,109,110], [111,112,113,114,115,116,117,118,119,120],[65,75,76,86,87,88,98,99,100,101],[10,11,12,13,23,24,25,35,36,46]]
        with open('Centers.txt', 'r') as file:
            for line in file:
                x = tuple(map(float, tuple(line.strip().split(" "))))
                self.locationspix.append(tuple(round(num) for num in x))
        self.goalzones = []
        self.indextoPlayerNum = dict()
        self.totalMovesMade = 0
        self.startzones = None
        self.boardState = ""
        
        self.indextoPlayerNum = dict()
        self.cantGoHere = []

        self.neighbors={
                0:[(1,3),(2,5)],
                1:[0,2,(3,6),(4,8)],
                2:[0,1,(4,7),(5,9)],
                3:[1,(4,5),(6,14),(7,16)],
                4:[1,2,3,5,(7,15),(8,17)],
                5:[(8,16),(9,18),(4,3),(2,0)],
                6:[(3,1),(7,8),(15,28),(14,26)],
                7:[3,(4,2),6,(8,9),(15,27),(16,29)],
                8:[(4,1),(7,6),5,9,(17,30),(16,28)],
                9:[(5,2),(8,7),(17,29),(18,31)],
                10:[(11,12),(23,35)],
                11:[10,(12,13),23,(24,36),35],
                12:[(11,10),(13,14),(24,35),(25,36)],
                13:[(12,11),(14,15),(25,36),(26,38)],
                14:[(13,12),(26,37),(27,39),(15,16),(6,3)],
                15:[6,(7,4),(14,13),(16,17),(27,38),(28,40)],
                16:[(7,3),(8,5),(15,14),(17,18),(28,39),(29,41)],
                17:[(8,4),9,(16,15),(18,19),(29,40),(30,42)],
                18:[(19,20),(30,41),(17,16),(31,43),(9,5)],
                19:[(18,17),(20,21),(31,42),(32,44)],
                20:[(19,18),(21,22),(32,43),(33,45)],
                21:[(20,19),22,(33,44),34],
                22:[(21,30),(34,45)],
                23:[10,11,(24,25),(35,36)],
                24:[11,12,23,(25,26),35,(36,47)],
                25:[12,13,(24,23),(26,27),(36,46),(37,48)],
                26:[13,(14,6),(25,24),(27,28),(37,47),(38,49)],
                27:[14,(15,7),(26,25),(28,39),(38,48),(39,50)],
                28:[(15,6),(16,8),(27,36),(29,30),(39,49),(40,51)],
                29:[(6,7),(17,9),(28,27),(30,31),(40,50),(41,52)],
                30:[(17,8),18,(29,28),(31,32),(41,51),(42,53)],
                31:[(18,9),19,(30,29),(32,33),(42,52),(43,54)],
                32:[20,19,(31,20),(33,34),(43,53),(44,55)],
                33:[21,20,(32,31),(34,35),(44,54),45],
                34:[22,21,(33,32),(34,35),(45,55)],
                35:[(23,10),(24,12),(36,37),(46,56)],
                36:[(24,11),(25,13),35,(37,38),46,(47,57)],
                37:[(25,12),(26,14),(36,35),(38,39),(47,56),(48,58)],
                38:[(26,13),(27,15),(37,36),(39,40),(48,57),(49,59)],
                39:[(27,14),(28,16),(38,37),(40,41),(49,58),(50,60)],
                40:[(28,15),(29,17),(39,38),(41,42),(50,59),(51,61)],
                41:[(29,16),(30,18),(40,39),(42,43),(51,60),(52,62)],
                42:[(30,17),(31,19),(41,40),(43,44),(52,61),(53,63)],
                43:[(31,18),(32,20),(42,41),(44,45),(53,62),(54,64)],
                44:[(32,19),(33,21),(43,42),45,(54,63),55],
                45:[(33,20),(34,22),(44,43),(55,64)],
                46:[(35,23),(36,25),(47,48),(56,66)],
                47:[(36,24),(37,26),(48,49),(57,67),46],
                48:[(37,25),(38,27),(49,50),(47,46),(57,66),(58,68)],
                49:[(38,26),(39,28),(50,51),(48,47),(58,67),(59,69)],
                50:[(39,27),(40,29),(51,52),(49,48),(59,68),(60,70)],
                51:[(40,28),(41,30),(52,53),(50,49),(60,69),(61,71)],
                52:[(41,29),(42,31),(53,54),(51,50),(61,70),(62,72)],
                53:[(42,30),(43,32),(54,55),(52,51),(62,71),(63,73)],
                54:[(43,31),(44,33),55,(53,52),(63,72),(64,74)],
                55:[(44,32),(45,34),(54,53),(64,73)],
                56:[(46,35),(47,37),(57,58),(66,77),(65,75)],
                57:[(47,36),(48,38),56,(58,59),(66,76),(67,78)],
                58:[(48,37),(49,38),(57,56),(58,59),(67,77),(68,79)],
                59:[(49,38),(50,39),(58,57),(59,60),(68,78),(69,80)],
                60:[(50,39),(51,41),(59,58),(61,62),(69,79),(70,81)],
                61:[(51,40),(52,42),(60,59),(62,63),(70,80),(71,82)],
                62:[(52,41),(53,43),(61,60),(63,64),(71,81),(72,83)],
                63:[(53,42),(54,44),(62,61),64,(71,81),(72,83)],
                64:[(54,43),(55,45),(63,62),(73,83),(74,88)],
                65:[(75,85),(76,88),(66,67),(56,47)],
                66:[(56,46),(57,48),(67,68),(76,87),(77,89),65],
                67:[(57,47),(55,49),(68,69),(77,88),(78,90),(66,65)],
                68:[(58,48),(56,50),(69,70),(78,89),(79,91),(67,66)],
                69:[(59,49),(57,51),(70,71),(79,90),(80,92),(68,67)],
                70:[(60,50),(58,52),(71,72),(80,91),(81,93),(69,68)],
                71:[(61,51),(59,53),(72,73),(81,92),(82,94),(70,69)],
                72:[(62,52),(60,54),(73,74),(82,93),(83,95),(71,70)],
                73:[(63,53),(64,55),74,(83,94),(84,96),(72,71)],
                74:[(64,54),(85,97),(84,95),(73,72)],
                75:[(65,56),(76,77),(86,98),(87,100)],
                76:[75,65,(66,57),(77,78),(88,101),(87,99)],
                77:[(76,75),(66,56),(67,58),(78,79),(89,102),(88,100)],
                78:[(77,76),(67,57),(68,59),(79,80),(90,103),(89,101)],
                79:[(78,77),(68,58),(69,60),(80,81),(91,104),(90,102)],
                80:[(79,78),(69,59),(70,61),(81,82),(92,105),(91,103)],
                81:[(80,79),(70,60),(71,62),(82,83),(93,106),(92,104)],
                82:[(81,80),(71,61),(72,63),(83,84),(94,107),(93,105)],
                83:[(82,81),(72,63),(73,64),(84,85),(95,108),(94,106)],
                84:[(83,82),(73,63),74,85,(96,109),(95,107)],
                85:[(74,64),(84,83),(97,110),(96,108)],
                86:[(75,65),(87,88),99,98],
                87:[75,(76,66),(88,89),86,99,100],
                88:[(76,65),(77,67),(89,90),101,100,(87,86)],
                89:[(77,66),(78,68),(90,91),101,(102,111),(88,87)],
                90:[(78,67),(79,69),(91,92),102,(103,112),(89,88)],
                91:[(79,68),(80,70),(92,93),(103,111),(104,113),(90,89)],
                92:[(80,69),(81,71),(93,94),(104,112),(105,114),(91,90)],
                93:[(81,70),(82,72),(94,95),(105,113),106,(92,91)],
                94:[(82,71),(83,73),(95,96),(106,114),107,(93,92)],
                95:[(83,72),(84,74),(96,97),107,108,(94,93)],
                96:[(84,73),85,97,108,109,(95,94)],
                97:[(85,74),109,110,(96,95)],
                98:[(86,75),(99,100)],
                99:[98,86,(87,76),(100,101)],
                100:[(99,98),(87,75),(88,77),(101,102)],
                101:[(100,99),(88,76),(89,78),(102,103)],
                102:[(101,100),(89,77),(90,79),(103,104),(111,115)],
                103:[(102,101),(90,78),(91,80),(104,105),(112,116),111],
                104:[(103,102),(91,79),(92,81),(105,106),(113,117),(112,115)],
                105:[(92,80),(93,82),(106,107),(104,103),(113,116),114],
                106:[(93,81),(94,83),(107,108),(105,104),(114,117)],
                107:[(94,82),(95,84),(108,109),(106,105)],
                108:[(95,83),(96,85),(109,110),(107,106)],
                109:[(96,84),97,110,(108,107)],
                110:[(97,85),(109,108)],
                111:[(102,89),(91,103),(113,112),(118,115)],
                112:[111,(103,90),(104,92),(113,114),(116,119),115],
                113:[(104,91),(105,93),114,117,(116,118),(112,111)],
                114:[(105,92),(106,94),(113,112),(117,119)],
                115:[(111,102),(112,104),(116,117),(118,120)],
                116:[(112,103),(113,105),117,115,118,119],
                117:[(114,106),(113,104),(116,115),(119,120)],
                118:[(115,111),(116,113),119,120],
                119:[(117,114),(116,112),118,120],
                120:[(118,115),(119,117)]
            }
        self.initializeBoard()
    def initializeBoard(self):
        if self.numPlayers == 2:
            self.boardState = self.players[0]*10 +"."*101 + self.players[1]*10
            self.goalzones = [self.allzones[3], self.allzones[0]]
            self.indextoPlayerNum[0] = (0,3)
            self.indextoPlayerNum[1] = (3,0)
            self.rotate = [3]
            self.cantGoHere = self.allzones[1] + self.allzones[2] + self.allzones[4] + self.allzones[5]
        elif self.numPlayers == 3:
            self.boardState = self.players[0]*10
            self.boardState += "."*(13+12+11+19)
            for i in range(4):
                self.boardState += self.players[2]*(1+i) + "."*(8-i) + self.players[1]*(1+i)
            self.boardState += "."*10
            self.goalzones = [self.allzones[3], self.allzones[5], self.allzones[1]]
            self.indextoPlayerNum[0] = (0,3)
            self.indextoPlayerNum[1] = (2,5)
            self.indextoPlayerNum[2] = (4,1)
            self.rotate = [2]
        elif self.numPlayers == 4:
            self.boardState = self.players[0]*10
            for i in range(4):
                self.boardState += "."*(9)
                self.boardState += self.players[1]*(4-i)
            self.boardState += "."*9
            for i in range(4):
                self.boardState += self.players[2]*(1+i)
                self.boardState += "."*(9)
            self.boardState += self.players[3]*10
            self.goalzones = [self.allzones[3], self.allzones[4], self.allzones[0], self.allzones[1]]
            self.indextoPlayerNum[0] = (0,3)
            self.indextoPlayerNum[1] = (1,4)
            self.indextoPlayerNum[2] = (3,0)
            self.indextoPlayerNum[3] = (4,1)
            self.rotate = [2,1]
        elif self.numPlayers == 6:
            self.boardState = self.players[0]*10 
            for i in range(4):
                self.boardState += self.players[5]*(4-i) + "."*(5+i) + self.players[1]*(4-i)
            self.boardState += "."*(9)
            for i in range(4):
                self.boardState += self.players[4]*(1+i) + "."*(8-i) + self.players[2]*(1+i)
            self.boardState += self.players[3]*10
            self.goalzones = [self.allzones[3], self.allzones[4], self.allzones[5], self.allzones[0], self.allzones[1], self.allzones[2]]
            for x in range(6):
                self.indextoPlayerNum[i] = (i,i + 3 if i <= 2 else i - 3)
            self.rotate = [1]
        temp = list()
        for x in self.boardState:
            # print(x)  
            temp.append(0 if x == "." else self.players.index(x) + 1)    
        self.state = np.array(temp)
        # print(self.state, self.boardState)

    def reset(self, seed=None, options=None):
        # print(self.players)
        self.__init__(arg=self.args)
        return (self.state, dict())

    def inMyEndzone(self, player, index):
        # print(self.playerToEnd)
        endzone = self.indextoPlayerNum[self.args.index(player)]
        return index in endzone
    def inWrongZone(self, index):
        if index in self.cantGoHere:
            return True
        return False
    def step(self, action):
        
        if self.isInvalidMove(action[0], action[1]):
            print("invalid:", action)
            return self.state, -100, False, False, dict()
        else:
            # cv2.waitKey(0)
            # print(action[0], action[1])
            # print("test", self.state, self.boardState)
            self.state[action[0]] = 0
            self.state[action[1]] = self.args.index(self.player) + 1
            boardList = list(self.boardState)
            # print(self.boardState, self.observation_space)
            boardList[action[0]] = "."
            boardList[action[1]] = self.player
            self.players.append(self.players.pop(0))
            # print(self.players)
            self.player = self.players[0]
            self.boardState = "".join(boardList)
            # self.rotateclockwise()
            # print(self.observation_space)
            # self.meanStartCenters[self.args.index(self.player)] = ((round(bestcenter[0]), round(bestcenter[1])), scoring[2])
            self.totalMovesMade += 1
            # self.boardState = "".join(boardList)
            # isDone = False
            if self.isWon():
                isDone = True
                reward = 100
            # else:
            #     print(self.players)
            self.render()
            reward = -1 if self.inMyEndzone(self.player, action[1]) else -2
            return self.state, reward, self.isWon(), False, dict()  #  return self.state, reward, done, info _
            # oCenter, oinertia = self.meanStartCenters(player) 
    def isInvalidMove(self, start, end):
        if end in self.cantGoHere:
            return True
        if self.boardState[end] != ".":
            # print("1st")
            return True
        if self.boardState[start] != self.player:
            # print("2nd")
            return True
        if start in self.goalzones[self.args.index(self.player)]:
            if not self.inMyEndzone(self.player, end):
                # print("3rd")
                return True
            else:
                return False
        # print(self.neighbors[start], end)
        return self.bfs(start, end)
        # return False
    # def rotateclockwise(self):
    #     template = [10,23,11,35,24,12,46,36,25,13,98,86,75,65,56,47,37,26,14,6,3,1,0,99,87,76,66,57,48,38,27,15,7,4,2,100,88,77,67,58,49,39,28,16,8,5,101,89,78,68,59,50,40,29,17,9,102,90,79,69,60,51,41,30,18,111,103,91,80,70,61,52,42,31,19,115,112,104,92,81,71,62,53,43,32,20,118,116,113,105,93,82,72,63,54,44,33,21,120,119,117,114,106,94,83,73,64,55,45,34,22,107,95,84,74,108,96,85,109,97,110]
    #     self.boardState = "".join([self.boardState[x] for x in template])
    def render(self):
        colors = {"B": (255,0,0), "R": (0,0,255), "G": (0,255,0), "W": (255,255,255), "Y":(0,255,255), "X":(0,0,0)}
        img = cv2.imread("final background.png") 
        # img = cv2.resize(img,(800,800))
        # print("length: ", len(bstate))
        for i, char in enumerate(self.boardState):
            if char in ["B", "G", "R", "W", "Y", "X"]:
                cv2.circle(img, self.locationspix[i], 18, colors[char], -1)
                # cv2.putText(img, str(i), self.locationspix[i], cv2.FONT_HERSHEY_SIMPLEX, 0.6
                #             , (0, 0, 0), 2)
        cv2.imshow("board", img)
        cv2.waitKey(1)
    def isWon(self):
        # print("moves:", self.moves)
        temp = [self.boardState[x] for x in self.goalzones[self.args.index(self.player)]]
        # print("test:", temp)
        return "." not in temp and self.player in temp

    def bfs(self, start, goal):
        queue = deque([start])
        visited = set()
        while queue:
            node = queue.popleft()
            if node == goal:
                return True
            if node not in visited:
                visited.add(node)
                for neighbor in self.neighbors[node]:
                    if type(neighbor) != int and neighbor[1] not in visited and neighbor[0] != "." and neighbor[1] == ".":
                        queue.extend(neighbor[1])
        return False

    # def sampleRandom(self):
    #     x = random.choice([i for i, c in enumerate(self.boardState) if c == self.player])
    #     temp = self.allPosMoves(x)
    #     # print(temp)
    #     return (x, random.choice(temp))

def main():
    Players = ["B", "G"]
    env = ChineseCheckersEnv(arg=Players)
    print(env.observation_space.sample())
    print(env.action_space.sample())
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000000)
    model.save("Second Attempt")
    # episodes = 1
    # for episode in range(1, episodes+1):
    #     state = env.reset()
    #     done = False
    #     score = 0
    #     # env.boardState = "B" * 121
    #     # env.render() 
        
    #     while not done:
            
    #         # action = env.action_space.sample()
    #         # print(action.type)
    #         # print(action, env.players)
    #         # print(env.boardState) 
    #         # print(env.player, env.boardState, len(env.boardState))
    #         temp = env.action_space.sample()
    #         # print(temp)
    #         env.render()
    #         # cv2.waitKey(0)
    #         obs, reward, done, truncated, info = env.step(temp)
    #         # print(obs, reward, done)
    #         # print(done)
            
    #         score+=reward
    #     cv2.destroyAllWindows()
    #     print('Episode:{} Score:{}'.format(episode, score))
        
        # check_env(env, warn=True)
    
    # obs = env.reset()[0]  # Reset the environment to get the initial observation
    # print(obs)
    # done = False
    # model = PPO.load("First Attempt.zip")
    # while not done:
    #     env.render()  # Optional: to visualize the environment
    #     # print(obs[0])
    #     action, _states = model.predict(obs, deterministic=True)  # Predict action
    #     obs, reward, done, info, moreinfo = env.step(action)  # Step in the environment
    #     print(f"Action: {action}, )Reward: {reward}, Done: {done}")
        
    # env.close()

if __name__ == "__main__":
    ...
    # main()