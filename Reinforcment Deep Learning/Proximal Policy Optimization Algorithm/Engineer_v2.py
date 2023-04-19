import random
from typing import List, Optional, Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import RenderFrame

import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray

from CatieAgent import CatieAgent

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

ALLOCATION_DICT = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}
DEALLOCATION_DICT = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}


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

        reward = 0
        done = False

        # Check constraints
        if (self.trial_number > 75 and self.assignments[0] < 100 - self.trial_number) \
                or (self.trial_number > 75 and self.assignments[1] < 100 - self.trial_number):
            reward = -100
            done = True
        # Check termination condition
        elif self.trial_number == self.max_trials:
            done = True
            reward = self.compute_reward()

        return observation, reward, done, {}

    def compute_reward(self) -> int:
        """
        Compute the reward for the agent based on the current state.

        Returns: Bias of all choices made by current CatieAgent
        """
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
                                *([nn.LazyLinear(hidden_size, device=DEVICE), activation_function()] * HIDDEN_LAYERS),
                                nn.Linear(hidden_size, output_size))

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

    def __init__(self, env, input_size: int, hidden_size: int, output_size: int, lr: float, epsilon: float,
                 activation_function: callable) -> None:
        """
        Initialize the Agent.
        Args:
            env: The environment in which the agent operates.
            input_size: The size of the input layer for the policy network.
            hidden_size: The size of the hidden layer for the policy network.
            output_size: The size of the output layer for the policy network.
            lr (float): The learning rate for the optimizer.
            epsilon (float): The exploration rate for the epsilon-greedy strategy.
            activation_function (callable): The activation function for the policy network.
        """
        self.env = env
        self.policy_net = PolicyNet(input_size, hidden_size, output_size, activation_function)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.max_fitness = 0
        self.epsilon = epsilon

    def select_action(self, state: Tuple[np.ndarray, Tuple[int, int], int]) -> torch.Tensor:
        """
        Select an action using the epsilon-greedy strategy.

        Args:
            state (tuple): A tuple containing three elements: catie_params (numpy array), assignments (tuple),
                           and trial (int).

        Returns:
            torch.Tensor: A tensor representing the constrained action probabilities.
        """
        assignments = state[9], state[10]  # State indices for assignments
        state_tensor = torch.Tensor(state)
        action_probs = self.policy_net(state_tensor)
        action_probs = nn.functional.softmax(action_probs, dim=-1)

        # Create a mask for valid actions - CONSTRAINTS
        mask = torch.FloatTensor([1.0, 1.0, 1.0, 1.0])
        add_mask = torch.FloatTensor([1e-8, 0.0, 0.0, 0.0])
        if assignments[0] >= 25:
            mask[1] = 0
            mask[3] = 0
        if assignments[1] >= 25:
            mask[2] = 0
            mask[3] = 0

        # Apply the mask to the action probabilities
        constrained_probs = ((action_probs + add_mask) * mask) / (((action_probs + add_mask) * mask).sum())

        # epsilon-greedy implementation
        if np.random.rand() < self.epsilon:
            valid_indices = np.argwhere(mask.numpy() > 0).flatten()
            random_action = np.random.choice(valid_indices)
            return constrained_probs * 0 + torch.eye(4)[random_action]

        return constrained_probs

    def train(self, n_episodes: int, n_rep: int, network_path: Optional[str] = None) -> None:
        """
        Train the agent using the REINFORCE algorithm.

        Args:
            n_episodes (int): The number of episodes to run the REINFORCE algorithm.
            n_rep (int): The number of iterations to run the CatieAgent on each policy.
                         Higher values result in less variance in the bias mean returned as a reward,
                         which in turn is used to calculate loss.
            network_path (str, optional): The file path to load the current network from. Defaults to None.
        """
        # INIT TRAINING SESSION
        self.load_network(network_path)
        episode_rewards = np.zeros(n_episodes)
        episode_rewards_std = np.zeros(n_episodes)
        episode_loss = np.zeros(n_episodes)
        rep_per_episode = np.arange(n_rep, n_rep + n_episodes) if INCREASING_REPS else n_rep
        prev_action_expectancy = torch.Tensor([.25, .25, .25, .25])
        # TRAINING LOOP
        for j in range(n_episodes):
            # Update the policy network
            rep_selected_action_probs = torch.zeros((rep_per_episode[j], N_TRIALS))
            rep_rewards = torch.zeros(rep_per_episode[j])

            # REPETITIONS FOR INCREASED ACCURACY IN RESULTS
            for repetition in range(rep_per_episode[j]):
                reward = 0
                state, _ = self.env.reset()
                selected_action_probs = torch.zeros(N_TRIALS)

                for trial in range(N_TRIALS):
                    action_prob = self.select_action(state)
                    action = ALLOCATION_DICT[torch.multinomial(action_prob, num_samples=1).item()]
                    selected_action_probs[trial] = action_prob[DEALLOCATION_DICT[action]]

                    next_state, reward, done, _ = self.env.step(action)

                    if done:
                        break
                    else:
                        state = next_state

                rep_selected_action_probs[repetition] = selected_action_probs
                rep_rewards[repetition] = reward

            # Calculate the episode return (discounted sum of rewards)
            G = torch.mean(rep_rewards[:j + 1])
            mean_selected_action_probs = torch.mean(rep_selected_action_probs, dim=0)
            # Add the MSE loss term to penalize the difference between G and the target value
            mse_loss = F.mse_loss(G, torch.tensor(N_TRIALS, dtype=torch.float64))

            action_expectancy = torch.sum(mean_selected_action_probs * torch.log(mean_selected_action_probs + 1e-6))
            loss = torch.sum(torch.log(mean_selected_action_probs/prev_action_expectancy))*mse_loss
            prev_action_expectancy = mean_selected_action_probs

            # Keep track of loss and rewards over time
            episode_rewards[j] = G
            episode_rewards_std[j] = torch.std(rep_rewards)
            episode_loss[j] = loss

            # Save highest fitness network so far
            if G > self.max_fitness:
                print(action_expectancy)
                self.max_fitness = G
                self.save_network(loss, rep_per_episode[j])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Print progress
            if j % 10 == 0:
                print(action_expectancy)
                self.plot_training_progress(episode_loss, episode_rewards, episode_rewards_std, j, loss)

    def load_network(self, network_path: Optional[str]) -> None:
        """
        Load the saved policy network state from the specified file path.

        Args:
            network_path (str, optional): The file path to the saved network state.
        """
        if network_path:
            with open(network_path, 'rb') as file:
                self.policy_net.load_state_dict(torch.load(file))

    def save_network(self, loss: float, cur_reps: int) -> None:
        """
        Save the current policy network state to a file.

        Args:
            loss (float): The loss value at the current episode.
            cur_reps (int): The number of repetitions per episode.
        """
        print(f"Top Network:{self.max_fitness} Loss:{loss} Iterations:{cur_reps}")
        with open(f'{HIDDEN_LAYERS}X{HIDDEN_SIZE}policy_net_bias{self.max_fitness}_reps{cur_reps}.pkl', 'wb') as file:
            torch.save(self.policy_net.state_dict(), file)

    @staticmethod
    def plot_training_progress(episode_loss: np.ndarray[float], episode_rewards: np.ndarray[float],
                               episode_rewards_std: np.ndarray[float], j: int, loss: torch.TensorType) -> None:
        """
        Plot the training progress of the agent, including episode rewards and episode loss, at every 100 episodes.

        This function generates a live plot of episode rewards and loss as the training progresses.
        The plot is updated every 100 episodes.

        Args:
            episode_loss (list): A list containing the loss values for each episode.
            episode_rewards (list): A list containing the mean rewards obtained for each episode.
            episode_rewards_std (list): A list containing the standard deviation of rewards for each episode.
            j (int): The current episode number.
            loss (float): The loss value of the current episode.

        Returns:
            None
        """
        print('Episode %d: reward = %d' % (j, episode_rewards[j]), loss)
        # Update the live plot
        x = range(len(episode_rewards))

        fig, ax1 = plt.subplots()

        # Plot episode_rewards with blue color
        ax1.set_ylabel('Episode Rewards', color='b')
        ax1.plot(x, episode_rewards, 'b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.errorbar(x, episode_rewards, yerr=episode_rewards_std, fmt='o', color='b', ecolor='b', capsize=2)

        # Create a second y-axis that shares the same x-axis
        ax2 = ax1.twinx()

        # Plot episode_loss with red color
        ax2.set_ylabel('Episode Loss', color='r')
        ax2.plot(x, episode_loss, 'r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Set x-axis label
        ax1.set_xlabel('Episode')

        plt.title('Episode Rewards and Loss')
        plt.show()

    def get_fitness(self) -> float:
        """
        Returns current highest bias mean achieved by model over N_REPETITIONS
        """
        return self.max_fitness


INPUT_SIZE = 12  # Nodes in input
HIDDEN_LAYERS = 2  # Number pf hidden layers
HIDDEN_SIZE = 20  # Number Nodes in input on hidden layers
OUTPUT_SIZE = 4  # Number of Nodes in output
LEARNING_RATE = 0.001  # Learning rate used for optimizer
EPSILON = 0.05  # for epsilon-greedy strategy
N_EPISODES = 500  # Total iterations of training cycle
N_REPETITIONS = 50  # Total repetitions per network to determine fittness
INCREASING_REPS = True  # Configuration whether repetitions increase as training progresses
N_TRIALS = 100  # Total trials done in each repetition
DEVICE = "cpu" if not torch.has_cuda else "cuda:0"
MAX_GRAD_NORM = 1.0

space = [
    Real(1e-6, 1e-2, name='learning_rate'),
    Integer(10, 200, name='hidden_size'),
    Real(1e-8, 1e-2, name='epsilon'),
]


@use_named_args(space)
def objective(**params) -> float:
    """
    Objective function for Bayesian optimization using skopt.

    This function trains an agent with the given hyper-parameters, evaluates its performance,
    and returns the negative mean reward as the optimization objective. The skopt library
    will try to minimize this objective by finding the best set of hyper-parameters.

    Args:
        **params (dict): A dictionary containing the hyper-parameters as keyword arguments.
            - learning_rate (float): The learning rate for the optimizer.
            - hidden_size (int): The number of hidden nodes in each hidden layer of the policy network.
            - epsilon (float): The epsilon value for the epsilon-greedy strategy.
            - activation_function (str): The name of the activation function to be used in the policy network.
              Possible values are 'relu', 'tanh', and 'leaky_relu'.

    Returns:
        float: The negative mean reward obtained after training the agent. This value is minimized during optimization.
    """
    env = CatieAgentEnv()
    print(params)

    # Create the agent with the given hyper-parameters
    agent = Agent(env, INPUT_SIZE, params['hidden_size'], OUTPUT_SIZE, params['learning_rate'], params['epsilon'],
                  nn.Tanh)

    # Train the agent
    agent.train(N_EPISODES, N_REPETITIONS)

    # Compute the optimization objective (e.g., negative mean reward)
    mean_reward = np.mean(agent.get_fitness())  # Use the appropriate variable from your code
    return -mean_reward


if __name__ == '__main__':
    torch.random.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    # Init plot
    plt.figure(figsize=(10, 5))

    # Optimizing over all hyper parameters
    result = gp_minimize(objective, space, n_calls=20, random_state=0, n_random_starts=5, acq_func='EI')
    best_params = result.x
    print("Best parameters found:")
    for i, param_name in enumerate(space):
        print(f"{param_name.name}: {best_params[i]}")

    # # Define hyper parameters
    # torch.autograd.set_detect_anomaly(True, check_nan=True)
    # env =
    # torch.random.manual_seed(1)
    # random.seed(1)
    # np.random.seed(1)
    #
    # # Create the environment and agent
    # agent = Agent(env, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, LEARNING_RATE, EPSILON)
    #
    # # Train the agent
    # agent.train(N_EPISODES, N_REPETITIONS)
