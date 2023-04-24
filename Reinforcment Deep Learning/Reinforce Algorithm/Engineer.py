import random
from typing import List, Optional, Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import RenderFrame

import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from tqdm import tqdm

from CatieAgent import CatieAgent

ALLOCATION_DICT = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}
DEALLOCATION_DICT = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
INPUT_SIZE = 12  # Nodes in input
HIDDEN_LAYERS = 2  # Number pf hidden layers
HIDDEN_SIZE = 20  # Number Nodes in input on hidden layers
OUTPUT_SIZE = 4  # Number of Nodes in output
LEARNING_RATE = 1e-3  # Learning rate used for optimizer
N_EPISODES = 1000001  # Total iterations of training cycle
EPISODES_PER_EPOCH = 1000
N_REPETITIONS = 700  # Total repetitions per network to determine fitness
INCREASING_REPS = False  # Configuration whether repetitions increase as training progresses
N_TRIALS = 100  # Total trials done in each repetition
DEVICE = "cpu" if not torch.has_cuda else "cuda:0"
MAX_GRAD_NORM = 1.0
EPOCHS_TO_PLOT = 25


class CatieAgentEnv(gym.Env):
    """
    Custom environment for the CatieAgent, which inherits from gym.Env.
    """

    def __init__(self, number_of_trials: int = 100) -> None:
        """
        Initialize the CatieAgentEnv.

        Args:
            number_of_trials (int): The number of trials to run. Defaults to 100.
        """
        self.agent = CatieAgent(number_of_trials=number_of_trials)
        self.action_space = spaces.MultiBinary(2)
        self.trial_number = 0
        self.max_trials = number_of_trials
        self.assignments = [0, 0]
        self.observation_space = spaces.Box(low=-np.ones(12), high=np.ones(12) * 4, dtype=np.float64)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> tuple[ndarray, dict]:
        """
        Reset the environment to its initial state.

        Args:
            seed (int, optional): A random seed for reproducibility. Defaults to None.
            options (dict, optional): Additional options for resetting. Defaults to None.

        Returns:
            observation (tuple): The initial observation of the environment after reset.
            info (dict): An empty dictionary.
        """
        super().reset(seed=seed)
        self.agent = CatieAgent()
        self.trial_number = 0
        self.assignments = [0, 0]
        observation = np.array([*self.agent.get_catie_param(), *self.assignments, self.trial_number], dtype=np.float64)
        return observation, {}

    def step(self, action: Tuple[int, int]) -> tuple[ndarray, int, bool, dict]:
        """
        Take a step in the environment based on the provided action.

        Args:
            action (tuple): A tuple representing the action to take.

        Returns:
            observation (tuple): The current observation of the environment after the action.
            reward (int): The reward for the taken action.
            done (bool): A flag indicating whether the environment has reached a terminal state.
            info (dict): An empty dictionary.
        """
        choice = self.agent.choose()
        self.agent.receive_outcome(choice, action)
        self.assignments[0] += action[0]
        self.assignments[1] += action[1]
        self.trial_number += 1

        observation = np.array([*self.agent.get_catie_param(), *self.assignments, self.trial_number], dtype=np.float64)
        reward = choice * N_TRIALS
        done = self.trial_number == self.max_trials

        return observation, reward, done, {}

    def compute_reward(self) -> int:
        """
        Compute the reward for the agent based on the current state.

        Returns: Bias of all choices made by current CatieAgent
        """
        if self.assignments[0] != 25 or self.assignments[1] != 25:
            return 0
        return self.agent.get_bias()

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return None


class PolicyNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, activation_function: callable):
        """
        Initialize the PolicyNet neural network.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The size of the hidden layers.
            output_size (int): The number of output features.
            activation_function (callable): The activation function to be used in the hidden layers.
        """
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, hidden_size),
                                activation_function(),
                                *([nn.LazyLinear(hidden_size, device=DEVICE),
                                   activation_function()] * HIDDEN_LAYERS),
                                nn.Linear(hidden_size, output_size), nn.Softmax(dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the PolicyNet neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        out = self.fc(x)
        return out


# Define the agent class
class Agent:
    """
    An Agent class that trains a reinforcement learning policy using the REINFORCE algorithm.
    """

    def __init__(self, env, input_size: int, hidden_size: int, output_size: int, lr: float,
                 activation_function: callable) -> None:
        """
        Initialize the Agent.
        Args:
            env: The environment in which the agent operates.
            input_size: The size of the input layer for the policy network.
            hidden_size: The size of the hidden layer for the policy network.
            output_size: The size of the output layer for the policy network.
            lr (float): The learning rate for the optimizer.
            activation_function (callable): The activation function for the policy network.
        """
        self.gamma = 0.99
        self.env = env
        self.policy_net = PolicyNet(input_size, hidden_size, output_size, activation_function)
        self.optimizer = optim.SGD(self.policy_net.parameters(), lr=lr)
        self.max_fitness = 0

    def select_action(self, state: Tuple[np.ndarray, Tuple[int, int], int]) -> torch.Tensor:
        """
        Select an action using the epsilon-greedy strategy.

        Args:
            state (tuple): A tuple containing three elements: catie_params (numpy array), assignments (tuple),
                           and trial (int).

        Returns:
            torch.Tensor: A tensor representing the constrained action probabilities.
        """
        assignments, trial_number = (state[9], state[10]), state[11]  # State indices for assignments
        action_probs = self.policy_net(torch.Tensor(state))

        # Create a mask for valid actions - CONSTRAINTS
        mask = torch.ones(4)
        add_mask = torch.FloatTensor([1e-9, 0.0, 0.0, 0.0])

        # Create masks for valid actions
        mask_0_condition = (trial_number > 75) & ((assignments[0] == 100 - trial_number) |
                                                  (assignments[1] == 100 - trial_number))
        mask_1_condition = assignments[0] < 25
        mask_2_condition = assignments[1] < 25

        mask[0] *= (~mask_0_condition)
        mask[1] *= mask_1_condition
        mask[2] *= mask_2_condition
        mask[3] *= (mask_1_condition & mask_2_condition)

        # Apply the mask to the action probabilities
        constrained_probs = (action_probs + add_mask) * mask
        constrained_probs /= constrained_probs.sum()

        return constrained_probs

    def train(self, n_episodes: int, network_path: Optional[str] = None) -> None:
        """
        Train the agent using the REINFORCE algorithm.

        Args:
            n_episodes (int): The number of episodes to run the REINFORCE algorithm.
            network_path (str, optional): The file path to load the current network from. Defaults to None.
        """
        # INIT TRAINING SESSION
        self.load_network(network_path)
        loss_mean = []
        # Preallocate tensors
        selected_action_probs = torch.zeros(N_TRIALS)
        episode_rewards = torch.zeros(N_TRIALS)
        # TRAINING LOOP
        for j in tqdm(range(n_episodes)):
            #  tensors
            selected_action_probs = selected_action_probs.detach() * 0
            episode_rewards = episode_rewards.detach() * 0

            # Initialize environment
            state, _ = self.env.reset()

            # Main loop
            for trial in range(N_TRIALS):
                action, action_prob = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                selected_action_probs[trial] = action_prob[DEALLOCATION_DICT[action]]
                episode_rewards[trial] = reward
                if done:
                    break
                else:
                    state = next_state

            # Compute loss and update parameters
            self.optimizer.zero_grad()
            log_probs = torch.log(selected_action_probs)
            loss = (-episode_rewards * log_probs).mean()

            loss_mean.append(loss.detach().numpy())

            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()

            if j % EPISODES_PER_EPOCH == 0:
                # Save network
                self.save_network()
                # Print Statistics
                self.plot_training_progress(j // EPISODES_PER_EPOCH, np.mean(loss_mean))
                loss_mean = []

    def get_action(self, state):
        action_prob = self.select_action(state)
        selected_index = torch.multinomial(action_prob, 1)
        action = ALLOCATION_DICT[int(selected_index)]
        return action, action_prob

    def load_network(self, network_path: Optional[str]) -> None:
        """
        Load the saved policy network state from the specified file path.

        Args:
            network_path (str, optional): The file path to the saved network state.
        """
        if network_path:
            with open(network_path, 'rb') as file:
                self.policy_net.load_state_dict(torch.load(file))

    def save_network(self) -> None:
        """
        Save the current policy network state to a file.
        """
        with open(f'2X{HIDDEN_SIZE}policy_net.pkl', 'wb') as file:
            torch.save(self.policy_net.state_dict(), file)

    def test_network(self, iterations):
        rep_rewards = np.zeros(iterations)
        # REPETITIONS FOR INCREASED ACCURACY IN RESULTS
        for repetition in range(iterations):
            state, _ = self.env.reset()
            for _ in range(N_TRIALS):
                action, action_prob = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)

                if done:
                    break
                else:
                    state = next_state
            rep_rewards[repetition] = self.env.compute_reward()
        return rep_rewards

    def plot_training_progress(self, j, loss) -> None:
        """
        Plot the training progress of the agent, including episode rewards and episode loss, at every 100 episodes.

        This function generates a live plot of episode rewards and loss as the training progresses.
        The plot is updated every 100 episodes.

        Args:
            j (int): The current episode number.

        Returns:
            None
        """
        global network_rewards_each_epoch
        global network_error_each_epoch
        global network_loss_each_epoch
        episode_rewards = self.test_network(N_REPETITIONS)
        network_rewards_each_epoch[j] = episode_rewards.mean()
        network_error_each_epoch[j] = np.std(episode_rewards) / np.sqrt(N_REPETITIONS)
        network_loss_each_epoch[j] = loss
        print(f"\nEpoch: {j}, Network mean Fitness: {network_rewards_each_epoch[j]}")
        print(f"Network Fitness error: {network_error_each_epoch[j]}")
        print(f"Network Loss: {network_loss_each_epoch[j]}")
        if j % EPOCHS_TO_PLOT == 0:
            # Update the live plot
            x = list(range(j + 1))
            # Plot episode_rewards with blue color
            figs, axes = plt.subplots(2)
            axes[0].set_ylabel('Epoch Rewards', color='b')
            axes[0].plot(x, network_rewards_each_epoch[:j + 1], 'b')
            axes[0].tick_params(axis='y', labelcolor='b')
            axes[0].errorbar(x, network_rewards_each_epoch[:j + 1], yerr=network_error_each_epoch[:j + 1], fmt='o',
                             color='b', ecolor='b', capsize=2)

            axes[1].set_ylabel('Epoch Loss', color='r')
            axes[1].plot(x, network_loss_each_epoch[:j + 1], color='r')
            axes[1].tick_params(axis='y', labelcolor='r')
            axes[1].set_xlabel('Epoch')
            plt.show()

    def get_fitness(self) -> float:
        """
        Returns current highest bias mean achieved by model over N_REPETITIONS
        """
        return self.max_fitness


if __name__ == '__main__':
    network_rewards_each_epoch = np.zeros(N_EPISODES // EPISODES_PER_EPOCH)
    network_error_each_epoch = np.zeros(N_EPISODES // EPISODES_PER_EPOCH)
    network_loss_each_epoch = np.zeros(N_EPISODES // EPISODES_PER_EPOCH)
    # Define hyper parameters
    env = CatieAgentEnv()
    torch.random.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    # Create the environment and agent
    agent = Agent(env, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, LEARNING_RATE, nn.ReLU)

    # Train the agent
    agent.train(N_EPISODES)
