import numpy as np
import tensorflow as tf
from tqdm import trange
from ChineseCheckers import ChineseCheckersBoard
from keras import layers, models

# === Game Settings ===
NUM_PLAYERS = 6
BOARD_SHAPE = (29, 19)
INPUT_DIM = BOARD_SHAPE[0] * BOARD_SHAPE[1] + 2
EPISODES = 1000
LEARNING_RATE = 1e-4


optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

