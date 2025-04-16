from ChineseCheckersBoard import ChineseCheckersBoard

env = ChineseCheckersBoard(2)
env.render(env.GlobalBoard)
print(env.allLegalActions(env.GlobalBoard, 1))
print(env.isGameOver(env.GlobalBoard, 1))

