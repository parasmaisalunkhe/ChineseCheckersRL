import gym
import numpy as np
import random
from collections import deque
from keras import layers, models, optimizers, losses
import tensorflow as tf

class HeuristicDQN(models.Model):
    def __init__(self, board_size):
        super(HeuristicDQN, self).__init__()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(256, activation='relu')
        self.output_layer = layers.Dense(1)  # Predict Q-value for (state, action)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)  # Shape: (batch_size, 1)

class DQNAgent:
    def __init__(self, env, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1,
                 learning_rate=1e-3, batch_size=64, buffer_size=10000):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = learning_rate
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

        self.board_size = env.observation_space['board'].shape[0]

        # Main and target networks
        self.model = HeuristicDQN(self.board_size)
        self.target_model = HeuristicDQN(self.board_size)
        self.target_model.set_weights(self.model.get_weights())

        self.optimizer = optimizers.Adam(self.lr)
        self.loss_fn = losses.MeanSquaredError()

    def get_action(self, obs):
        board = obs["board"]
        legal_moves = obs["action_mask"]

        if np.random.rand() < self.epsilon:
            return random.choice(legal_moves)

        scores = []
        for move in legal_moves:
            from_pos, to_pos = move
            next_board = self.simulate_move(board, from_pos, to_pos)
            input_vector = np.concatenate([next_board, [from_pos, to_pos]]).astype(np.float32)
            score = self.model(tf.expand_dims(input_vector, axis=0)).numpy()[0][0]
            scores.append(score)

        best_move_idx = np.argmax(scores)
        return legal_moves[best_move_idx]

    def simulate_move(self, board, from_pos, to_pos):
        board = np.copy(board)
        board[to_pos] = board[from_pos]
        board[from_pos] = 0.0
        return board

    def remember(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.buffer) < self.batch_size:
            return

        minibatch = random.sample(self.buffer, self.batch_size)

        inputs = []
        targets = []

        for state, action, reward, next_state, done in minibatch:
            state_board = state["board"]
            from_pos, to_pos = action
            input_vec = np.concatenate([state_board, [from_pos, to_pos]]).astype(np.float32)

            if done or not next_state["action_mask"]:
                target = reward
            else:
                q_values = []
                for move in next_state["action_mask"]:
                    n_from, n_to = move
                    next_board = self.simulate_move(next_state["board"], n_from, n_to)
                    next_input = np.concatenate([next_board, [n_from, n_to]]).astype(np.float32)
                    q_val = self.target_model(tf.expand_dims(next_input, axis=0)).numpy()[0][0]
                    q_values.append(q_val)
                target = reward + self.gamma * np.max(q_values)

            inputs.append(input_vec)
            targets.append(target)

        inputs_tensor = tf.convert_to_tensor(inputs)
        targets_tensor = tf.convert_to_tensor(targets)

        with tf.GradientTape() as tape:
            predictions = tf.squeeze(self.model(inputs_tensor), axis=1)
            loss = self.loss_fn(targets_tensor, predictions)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
