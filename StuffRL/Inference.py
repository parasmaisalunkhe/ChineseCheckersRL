import numpy as np
import tensorflow as tf
import time
from keras import layers, models
from TestingGymEnvMore import ChineseCheckersBoard  # Your custom environment
NUM_PLAYERS = 2
BOARD_SHAPE = (29, 19)
INPUT_DIM = BOARD_SHAPE[0] * BOARD_SHAPE[1] + 2
# Load saved model
def create_q_model():
    model = models.Sequential([
        layers.Input(shape=(INPUT_DIM,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # Q-value
    ])
    return model

q_model = create_q_model()
q_model = models.load_model("q_model_chinese_checkers_1.h5")
print("Q-model loaded for inference.")

# Start a game
env = ChineseCheckersBoard(NUM_PLAYERS)
board, _ = env.reset()

done = False

while not done:
    state = board["obs"].flatten().astype(np.float32)
    legal_moves = board["action_mask"]

    if not legal_moves:
        break

    # Prepare input batch
    obs_batch = np.repeat(state[None, :], len(legal_moves), axis=0)
    move_batch = np.array(legal_moves, dtype=np.float32)
    input_batch = np.concatenate([obs_batch, move_batch], axis=1)

    # Predict Q-values and choose best move
    q_values = q_model.predict(input_batch, verbose=0).squeeze()
    best_index = np.argmax(q_values)
    best_move = legal_moves[best_index]

    # print(f"Best move selected: {best_move} | Q-value: {q_values[best_index]:.4f}")
    env.render(env.GlobalBoard)
    time.sleep(0.3)
    print("\033[H\033[J")
    board, _, done, _, _ = env.step(best_move)
