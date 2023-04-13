from abc import ABC

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.spaces import Tuple

from CatieAgent import CatieAgent

ALLOCATION_DICT = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}


class CatieAgentEnv(gym.Env):
    """
    ## Description

    This environment corresponds to the version of the CATIE choice algorithm described by Ori Plonsky in
    "Plonsky,O.,&Erev,I.(2017). Learning in Settings with Partial Feedback and the Wavy Recency Effect of Rare Events"

    ## Action Space
    All different reward allocations

    ## Observation Space
    The observation is a `ndarray`,int tuple where that array is of size (n_trials,2) and the integer represents the
    current trial number.

    ## Rewards
    At the end of each experiment the total bias is calculated
    """

    def __init__(self, number_of_trials=100):
        self.agent = CatieAgent(number_of_trials=number_of_trials)
        # For each of the 4 choices made by the engineer.
        self.action_space = spaces.MultiBinary(2)
        # Current trial number
        self.trial_number = 0
        self.max_trials = number_of_trials
        self.assignments = [0, 0]
        self.observation_space = spaces.Tuple((spaces.MultiDiscrete(np.ones(number_of_trials) * 3),
                                               spaces.Discrete(self.max_trials)))

    def reset(self, *, seed=None, options=None, ):
        self.agent = CatieAgent()
        self.trial_number = 0
        self.assignments = [0, 0]
        observation = self.agent.get_catie_param(), self.assignments, self.trial_number
        return observation, {}

    def step(self, action):
        choice = self.agent.choose()
        self.agent.receive_outcome(choice, action)
        self.assignments[0] += action[0]
        self.assignments[1] += action[1]
        self.trial_number += 1

        observation = self.agent.get_catie_param(), self.assignments, self.trial_number

        # Check constraints
        if (self.trial_number > 75 and self.assignments[0] < 100 - self.trial_number) \
                or (self.trial_number > 75 and self.assignments[1] < 100 - self.trial_number):
            reward = self.trial_number - 100
            done = True
        # Check termination condition
        else:
            done = (self.trial_number == self.max_trials)
            reward = self.compute_reward() if done else 0
        return observation, reward, done, {}

    def compute_reward(self):
        return self.agent.get_bias()
