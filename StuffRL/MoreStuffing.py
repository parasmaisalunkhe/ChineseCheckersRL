import numpy as np
import tensorflow as tf
from collections import deque
import random

class ValueNetwork(tf.keras.Model):
    def __init__(self, board_size):
        super(ValueNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1)  # Predict value of board

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

class LookaheadRLAgent:
    def __init__(self, board_size, learning_rate=1e-3, gamma=0.99):
        self.value_net = ValueNetwork(board_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.gamma = gamma
        self.memory = deque(maxlen=10000)
        self.batch_size = 64

    def select_move(self, env, player_id):
        legal_moves = env.get_legal_moves(player_id)
        best_value = -np.inf
        best_move = None

        for move in legal_moves:
            sim_env = env.copy()
            sim_env.apply_move(player_id, move)
            obs = sim_env.get_observation(player_id)
            obs_tensor = tf.convert_to_tensor(obs[np.newaxis, :], dtype=tf.float32)
            predicted_value = self.value_net(obs_tensor).numpy()[0][0]

            if predicted_value > best_value:
                best_value = predicted_value
                best_move = move

        return best_move

    def store_transition(self, obs, reward, next_obs, done):
        self.memory.append((obs, reward, next_obs, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        obs_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)

        obs_batch = tf.convert_to_tensor(obs_batch, dtype=tf.float32)
        next_obs_batch = tf.convert_to_tensor(next_obs_batch, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        done_batch = tf.convert_to_tensor(done_batch, dtype=tf.float32)

        next_values = tf.squeeze(self.value_net(next_obs_batch), axis=1)
        target_values = reward_batch + self.gamma * next_values * (1.0 - done_batch)

        with tf.GradientTape() as tape:
            predicted_values = tf.squeeze(self.value_net(obs_batch), axis=1)
            loss = tf.keras.losses.MSE(target_values, predicted_values)

        grads = tape.gradient(loss, self.value_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.value_net.trainable_variables))
        return loss.numpy()

