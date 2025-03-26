#rock paper scissors
import gymnasium as gym

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class RockPaperScissors(MultiAgentEnv):
    """Two-player environment for the famous rock paper scissors game.
    Both players always move simultaneously over a course of 10 timesteps in total.
    The winner of each timestep receives reward of +1, the losing player -1.0.

    The observation of each player is the last opponent action.
    """

    ROCK = 0
    PAPER = 1
    SCISSORS = 2
    LIZARD = 3
    SPOCK = 4

    WIN_MATRIX = {
        (ROCK, ROCK): (0, 0),
        (ROCK, PAPER): (-1, 1),
        (ROCK, SCISSORS): (1, -1),
        (PAPER, ROCK): (1, -1),
        (PAPER, PAPER): (0, 0),
        (PAPER, SCISSORS): (-1, 1),
        (SCISSORS, ROCK): (-1, 1),
        (SCISSORS, PAPER): (1, -1),
        (SCISSORS, SCISSORS): (0, 0),
    }
    def __init__(self, config=None):
        super().__init__()

        self.agents = self.possible_agents = ["player1", "player2"]

        # The observations are always the last taken actions. Hence observation- and
        # action spaces are identical.
        self.observation_spaces = self.action_spaces = {
            "player1": gym.spaces.Discrete(3),
            "player2": gym.spaces.Discrete(3),
        }
        self.last_move = None
        self.num_moves = 0
    def reset(self, *, seed=None, options=None):
        self.num_moves = 0

        # The first observation should not matter (none of the agents has moved yet).
        # Set them to 0.
        return {
            "player1": 0,
            "player2": 0,
        }, {}  # <- empty infos dict
    def step(self, action_dict):
        self.num_moves += 1

        move1 = action_dict["player1"]
        move2 = action_dict["player2"]

        # Set the next observations (simply use the other player's action).
        # Note that because we are publishing both players in the observations dict,
        # we expect both players to act in the next `step()` (simultaneous stepping).
        observations = {"player1": move2, "player2": move1}

        # Compute rewards for each player based on the win-matrix.
        r1, r2 = self.WIN_MATRIX[move1, move2]
        rewards = {"player1": r1, "player2": r2}

        # Terminate the entire episode (for all agents) once 10 moves have been made.
        terminateds = {"__all__": self.num_moves >= 10}

        # Leave truncateds and infos empty.
        return observations, rewards, terminateds, {}, {}

from ray.rllib.algorithms.ppo import PPOConfig

# Create a config instance for the PPO algorithm.
config = (
    PPOConfig()
    .environment("Pendulum-v1")
)