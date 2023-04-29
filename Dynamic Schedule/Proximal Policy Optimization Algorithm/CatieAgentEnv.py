from typing import List, Optional, Dict, Union
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import RenderFrame
import numpy as np
from numpy import ndarray
from CatieAgent import CatieAgent


class CatieAgentEnv(gym.Env):
    """
    Custom gymnasium environment for the CatieAgent, which inherits from gym.Env.
    """

    def __init__(self, number_of_trials: int = 100) -> None:
        """
        Initialize the CatieAgentEnv.

        Args:
            number_of_trials (int): The number of trials to run. Defaults to 100.
        """
        self.agent = CatieAgent(number_of_trials=number_of_trials)
        self.action_space = spaces.Box(low=np.zeros(4), high=np.ones(4), dtype=np.float64)
        self.trial_number = 0
        self.max_trials = number_of_trials
        self.assignments = [0, 0]
        self.assignments_left_ratio = [1, 1]
        self.observation_space = spaces.Box(low=-np.ones(14), high=np.ones(14) * 4, dtype=np.float64)

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
        self.assignments_left_ratio = [1, 1]
        observation = np.array(
            [*self.agent.get_catie_param(), *self.assignments, *self.assignments_left_ratio, self.trial_number],
            dtype=np.float64)
        return observation, {}

    def step(self, action):
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

        self.assignments_left_ratio = [(25 - self.assignments[0]) / self.trial_number,
                                       (25 - self.assignments[1]) / self.trial_number]

        observation = np.array(
            [*self.agent.get_catie_param(), *self.assignments, *self.assignments_left_ratio, self.trial_number],
            dtype=np.float64)
        reward = choice
        done = self.trial_number == self.max_trials

        return observation, reward, done, {}

    def compute_reward(self) -> int:
        """
        Compute the reward for the agent based on the current state.

        Returns: Bias of all choices made by current CatieAgent
        """
        return self.agent.get_bias()

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return None
