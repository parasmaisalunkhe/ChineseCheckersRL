import numpy as np
import tensorflow as tf
from keras import layers, models
from tqdm import trange
import matplotlib.pyplot as plt
from TestingGymEnvMore import ChineseCheckersBoard

# === Hyperparameters ===
BOARD_SHAPE = (29, 19)
MOVE_SHAPE = (2, 1)
STATE_DIM = BOARD_SHAPE[0] * BOARD_SHAPE[1]  # Flattened board state
MOVE_DIM = MOVE_SHAPE[0] * MOVE_SHAPE[1]     # Flattened move representation
INPUT_DIM = STATE_DIM + MOVE_DIM             # Combined input dimension
EPISODES = 1000
LEARNING_RATE = 1e-4
GAMMA = 0.95                  # Discount factor
EPSILON_START = 1.0           # Initial exploration rate
EPSILON_END = 0.01            # Minimum exploration rate
EPSILON_DECAY = 0.995         # Decay rate per episode
BATCH_SIZE = 64               # Batch size for training
MEMORY_SIZE = 10000           # Replay memory size
TARGET_UPDATE_FREQ = 10       # How often to update target network

# === Q-Network Architecture ===
def create_q_network():
    """Creates a neural network that takes state and move as input and outputs Q-value"""
    # Input layers for state and move
    state_input = layers.Input(shape=(STATE_DIM,), name='state_input')
    move_input = layers.Input(shape=(MOVE_DIM,), name='move_input')
    
    # Process state with a deeper network
    state_stream = layers.Dense(512, activation='relu')(state_input)
    state_stream = layers.Dense(256, activation='relu')(state_stream)
    state_stream = layers.Dense(128, activation='relu')(state_stream)
    
    # Process move
    move_stream = layers.Dense(64, activation='relu')(move_input)
    move_stream = layers.Dense(32, activation='relu')(move_stream)
    
    # Combine state and move streams
    combined = layers.concatenate([state_stream, move_stream])
    combined = layers.Dense(128, activation='relu')(combined)
    combined = layers.Dense(64, activation='relu')(combined)
    
    # Output Q-value
    q_value = layers.Dense(1, activation='linear')(combined)
    
    model = models.Model(inputs=[state_input, move_input], outputs=q_value)
    return model

# === Replay Memory ===
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, state, move, reward, next_state, next_legal_moves, done):
        """Saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, move, reward, next_state, next_legal_moves, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Samples a batch of transitions"""
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        states, moves, rewards, next_states, next_legal_moves, dones = zip(*batch)
        return np.array(states), np.array(moves), np.array(rewards), np.array(next_states), next_legal_moves, np.array(dones)
    
    def __len__(self):
        return len(self.memory)

# === Training Setup ===
q_network = create_q_network()
target_network = create_q_network()
target_network.set_weights(q_network.get_weights())

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_fn = tf.keras.losses.MeanSquaredError()

memory = ReplayMemory(MEMORY_SIZE)
epsilon = EPSILON_START

# === Move Selection ===
def select_move(state, legal_moves, q_network, epsilon):
    """
    Selects a move using epsilon-greedy policy
    Args:
        state: Current board state (flattened)
        legal_moves: List of legal moves (each move is (2,1))
        q_network: Q-network for value estimation
        epsilon: Exploration rate
    Returns:
        Selected move and the corresponding (state, move) pair
    """
    state = np.array(state, dtype=np.float32).flatten()
    
    if np.random.rand() < epsilon:
        # Random exploration
        idx = np.random.choice(len(legal_moves))
        selected_move = legal_moves[idx]
    else:
        # Greedy action selection
        # Evaluate all legal moves
        state_batch = np.tile(state, (len(legal_moves), 1))
        moves_batch = np.array(legal_moves).reshape(-1, MOVE_DIM)
        
        q_values = q_network.predict([state_batch, moves_batch], verbose=0).flatten()
        best_idx = np.argmax(q_values)
        selected_move = legal_moves[best_idx]
    
    return selected_move, (state, selected_move.reshape(MOVE_DIM))

# === Training Loop ===
episode_rewards = []
episode_losses = []
epsilon_history = []

for episode in range(EPISODES):
    env = ChineseCheckersBoard(2)
    obs, info = env.reset()
    state = obs["obs"].flatten()
    legal_moves = obs["action_mask"]
    
    total_reward = 0
    done = False
    losses = []
    
    while not done and env.num_moves < 1000:
        # Select and execute action
        move, (current_state, current_move) = select_move(state, legal_moves, q_network, epsilon)
        
        obs, reward, done, truncated, info = env.step(move)
        next_state = obs["obs"].flatten()
        next_legal_moves = obs["action_mask"]
        
        # Store transition in replay memory
        memory.push(current_state, current_move, reward, next_state, next_legal_moves, done)
        
        # Sample batch from memory and train
        if len(memory) >= BATCH_SIZE:
            states, moves, rewards, next_states, next_legal_moves_list, dones = memory.sample(BATCH_SIZE)
            
            # Compute target Q-values using target network
            max_next_q_values = np.zeros(BATCH_SIZE)
            
            # For states where the game isn't done, compute max Q for next state
            not_done_indices = np.where(dones == False)[0]
            
            if len(not_done_indices) > 0:
                # Process these states in batches to avoid memory issues
                batch_size = 32  # Smaller batch size for prediction
                for i in range(0, len(not_done_indices), batch_size):
                    batch_indices = not_done_indices[i:i+batch_size]
                    batch_next_states = next_states[batch_indices]
                    batch_next_legal_moves = [next_legal_moves_list[idx] for idx in batch_indices]
                    
                    # Prepare inputs for all legal moves for each state in batch
                    state_inputs = []
                    move_inputs = []
                    for state_idx, (ns, nlms) in enumerate(zip(batch_next_states, batch_next_legal_moves)):
                        if len(nlms) > 0:
                            state_inputs.extend([ns] * len(nlms))
                            move_inputs.extend([m.reshape(MOVE_DIM) for m in nlms])
                    
                    if len(state_inputs) > 0:
                        state_inputs = np.array(state_inputs)
                        move_inputs = np.array(move_inputs)
                        
                        # Predict Q-values for all possible next moves
                        next_q_values = target_network.predict(
                            [state_inputs, move_inputs], 
                            verbose=0
                        ).flatten()
                        
                        # Find max Q-value for each next state
                        ptr = 0
                        for state_idx, idx in enumerate(batch_indices):
                            num_moves = len(batch_next_legal_moves[state_idx])
                            if num_moves > 0:
                                max_next_q_values[idx] = np.max(next_q_values[ptr:ptr+num_moves])
                                ptr += num_moves
            
            targets = rewards + GAMMA * max_next_q_values * (1 - dones)
            
            # Train Q-network
            with tf.GradientTape() as tape:
                current_q_values = tf.squeeze(q_network([states, moves], training=True))
                loss = loss_fn(targets, current_q_values)
            grads = tape.gradient(loss, q_network.trainable_variables)
            optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
            losses.append(loss.numpy())
        
        state = next_state
        legal_moves = next_legal_moves
        total_reward += reward
    
    # Decay epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    
    # Log metrics
    episode_rewards.append(total_reward)
    episode_losses.append(np.mean(losses) if losses else 0)
    epsilon_history.append(epsilon)
    
    # Update target network
    if episode % TARGET_UPDATE_FREQ == 0:
        target_network.set_weights(q_network.get_weights())
    
    # Print progress
    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {total_reward:.2f}, Loss: {np.mean(losses):.4f}, Epsilon: {epsilon:.3f}")

# === Save Model ===
q_network.save("board_game_q_network.h5")
print("Model saved successfully.")

# === Plot Results ===
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(episode_rewards)
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")

plt.subplot(1, 2, 2)
plt.plot(episode_losses)
plt.title("Training Loss")
plt.xlabel("Episode")
plt.ylabel("Loss")

plt.tight_layout()
plt.savefig("training_results.png")
plt.show()