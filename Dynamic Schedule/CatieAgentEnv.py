import numpy as np
from numpy import ndarray
from CatieAgent import CatieAgent


class CatieAgentEnv:
    """
    Custom environment for the CatieAgent, which inherits from gym.Env.
    """

    def __init__(self, beta=0, number_of_trials: int = 100) -> None:
        """
        Initialize the CatieAgentEnv.

        Args:
            number_of_trials (int): The number of trials to run. Defaults to 100.
        """
        self.beta = beta
        self.agent = CatieAgent(number_of_trials=number_of_trials)
        self.action_space = np.zeros(4)
        self.trial_number = 0
        self.max_trials = number_of_trials
        self.assignments = [0, 0]
        self.assignments_left_ratio = [1, 1]
        self.observation_space = np.zeros(14)

    def reset(self, beta=0) -> tuple[ndarray, dict]:
        """
        Reset the environment to its initial state.

        Args:
            seed (int, optional): A random seed for reproducibility. Defaults to None.
            options (dict, optional): Additional options for resetting. Defaults to None.
            :param beta:

        Returns:
            observation (tuple): The initial observation of the environment after reset.
            info (dict): An empty dictionary.
        """
        self.agent = CatieAgent()
        self.trial_number = 0
        self.assignments = [0, 0]
        self.assignments_left_ratio = [1, 1]
        self.beta = beta
        observation = np.array(
            [*self.agent.get_catie_param(), 0, 0, *self.assignments_left_ratio,
             0],
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

        self.assignments_left_ratio = [(25 - self.assignments[0]) / 25,
                                       (25 - self.assignments[1]) / 25]

        observation = np.array(
            [*self.agent.get_catie_param(), self.assignments[0] / 25,
             self.assignments[1] / 25,
             *self.assignments_left_ratio, self.trial_number / 100],
            dtype=np.float64)
        reward = 0
        if self.trial_number == self.max_trials:
            proximity_constraint = ((25 - self.assignments[0]) ** 2) + \
                                   ((25 - self.assignments[1]) ** 2)
            reward = self.agent.get_bias() - (self.beta * proximity_constraint)
        if reward > 100:
            print(self.agent.get_bias())
            print(self.beta)
        done = self.trial_number == self.max_trials
        return observation, reward, done, choice

    def compute_reward(self) -> int:
        """
        Compute the reward for the agent based on the current state.

        Returns: Bias of all choices made by current CatieAgent
        """
        return self.agent.get_bias()
