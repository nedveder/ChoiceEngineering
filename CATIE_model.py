import statistics

import numpy as np
from numpy import float64

TAO = 0.29
EPSILON = 0.30
PHI = 0.71
K = 2
ALTERNATIVE_A = 1
ALTERNATIVE_B = 0


class CatieAgent:

    def __init__(self):
        self.alt_a_contingencies = dict()
        self.alt_b_contingencies = dict()
        # List of previous choices, where 1's indicate a
        # choice of alternative A and 0's indicate choice of alternative B.
        self.previous_choices = []
        self.outcomes = []  # List of all outcomes
        self.alt_a_outcomes = []  # List of outcomes for Alternative A on each trial that A was chosen
        self.alt_b_outcomes = []  # List of outcomes for Alternative B on each trial that B was chosen
        self.trial_number = 0  # Index of current trial (equals len(self.previous_choices)
        self.ca_alt_1 = 0
        self.ca_alt_2 = 0
        self.surprises = []
        self.k = np.random.randint(0, K + 1)
        self.choice_type_history = ""

    def choose(self):
        """
        Controls choice mechanism for CATIE agent, using the 4 different modes for decision-making.
            1.If the agent chose the same alternative in two consecutive trials and their outcomes differed, it
            would choose in the subsequent trial the trend (or heuristic) mode with a probability 𝜏.
            2. If the trend mode was not chosen, the agent would choose the explore mode with a probability p_explore.
            3. If neither trend nor explore modes were chosen,the agent would choose, the inertia mode with
             a probability 𝜙.
            4. If none of the above modes was chosen, the agent would choose the contingent average mode.
        Adds the current choice made to the prev_choice list and returns the choice to caller.
        The method updates a list containing the different modes used for every decision.
        """

        # INIT MODE
        # For the first two trials we sample both options in random order.
        if self.trial_number < 2:
            # Samples both alternatives with random order
            choice = self.random_choice() if self.trial_number == 0 else self.other_choice()
            self.choice_type_history += "0"

        # TREND MODE
        # If we chose the same alternative in the past two trials and received different outcomes then with probability
        # TAO we enter the TREND mode.
        elif self.previous_choices[self.trial_number - 1] == self.previous_choices[self.trial_number - 2] \
                and self.outcomes[self.trial_number - 1] != self.outcomes[self.trial_number - 2] \
                and np.random.random() <= TAO:
            choice = self.trend()
            self.choice_type_history += "T"

        # EXPLORE MODE
        # Explore is entered with probability p_explore which is determined using calculate_p_explore method at
        # each trial. Explore mode means choosing at random from both alternatives.
        elif np.random.random() <= self.calculate_p_explore():
            choice = self.random_choice()
            self.choice_type_history += "E"

        # INERTIA MODE
        # Choose the previous choice again.
        elif np.random.random() <= PHI:
            choice = self.previous_choices[-1]
            self.choice_type_history += "I"

        # CONTINGENT AVERAGE MODE
        # In the contingent average (CA) mode, the agent chooses the alternative associated with the higher k-CA,
        # defined as the average payoff observed in all previous trials which followed the same sequence of k outcomes
        # See contingent_average method for further information.
        else:
            ca_a = self.contingent_average(self.k, ALTERNATIVE_A)
            ca_b = self.contingent_average(self.k, ALTERNATIVE_B)

            if ca_a == ca_b:
                choice = self.random_choice()
            else:
                choice = ALTERNATIVE_A if ca_a > ca_b else ALTERNATIVE_B
            self.choice_type_history += "A"

        return choice

    @staticmethod
    def random_choice():
        """
        :return: A random choice between the two alternatives.
        """
        return np.random.choice([ALTERNATIVE_A, ALTERNATIVE_B])

    def other_choice(self):
        """
        :return: Given the previous choice return the other alternative.
        """
        return (self.previous_choices[-1] + 1) % 2

    def trend(self):
        """
        In this mode the agent chooses the “positive trend” meaning if the outcome of the previous choice was
        bigger then the one preceding it, we choose it ̧otherwise we switch choices.
        :return: The choice corresponding with TREND mode.
        """
        matching_choice_rewards = self.alt_a_outcomes if self.previous_choices[-1] == 1 else self.alt_b_outcomes
        if matching_choice_rewards[- 1] > matching_choice_rewards[- 2]:
            return self.previous_choices[-1]
        else:
            return self.other_choice()

    def contingent_average(self, k, choice):
        """
        The contingent average of the input alternative.
        """
        if k == 0:
            choice_outcomes = self.alt_a_outcomes if choice == ALTERNATIVE_A else self.alt_b_outcomes
            return [np.mean(choice_outcomes)]
        # Get last k choices and outcomes
        current_contingency = tuple(zip(self.previous_choices[-self.k:], self.outcomes[-self.k:]))
        # Get appropriate contingency dictionary
        choice_contingencies = self.alt_a_contingencies if choice == ALTERNATIVE_A else self.alt_b_contingencies
        # If contingency already exists in dictionary then return it
        if current_contingency in choice_contingencies:
            return [np.mean(choice_contingencies[current_contingency])]
        # Current contingency does not exist
        if not choice_contingencies:  # If there are no k-length contingencies simply consider the mean
            choice_outcomes = self.alt_a_outcomes if choice == ALTERNATIVE_A else self.alt_b_outcomes
            return [np.mean(choice_outcomes)]
        else:
            # The current contingency does not appear, but other contingencies do. In such case the user gets
            # "confused" and chooses one of the other contingencies at random. Since the current
            # implementation ultimately computes probability, this scenario return all the possible outcomes
            # of the confusion (which are by definition chosen with uniform probability). Namely, the average
            # of each existing contingent average outcome
            return [np.mean(confused_contingency_outcomes) for confused_contingency_outcomes in
                    choice_contingencies.values()]

    def receive_outcome(self, choice, outcome):
        if choice == ALTERNATIVE_A:
            self.alt_a_outcomes.append(outcome[choice])
            obs_sd = 0 if self.trial_number < 2 else np.std(self.alt_a_outcomes, dtype=float64)
        else:  # choice == ALTERNATIVE_B
            self.alt_b_outcomes.append(outcome[choice])
            obs_sd = 0 if self.trial_number < 2 else np.std(self.alt_b_outcomes, dtype=float64)

        if obs_sd:  # If standard deviation in positive
            exp_t_i = np.mean(self.contingent_average(self.k, choice))  # Expected contingent average for choice
            expected_actual_reward_diff = abs(exp_t_i - outcome[choice])
            surprise_t = expected_actual_reward_diff / (obs_sd + expected_actual_reward_diff)
        else:
            surprise_t = 0

        self.surprises.append(surprise_t)
        self.previous_choices.append(choice)
        self.outcomes.append(outcome[choice])

        # Update contingent dictionary
        if 0 < self.k < self.trial_number:
            # Get previous k contingencies (before current trial)
            current_contingency = tuple(zip(self.previous_choices[-self.k - 1:-1], self.outcomes[-self.k - 1:-1]))
            choice_contingencies = self.alt_a_contingencies if choice == ALTERNATIVE_A else self.alt_b_contingencies
            if current_contingency not in choice_contingencies:
                choice_contingencies[current_contingency] = []
            choice_contingencies[current_contingency].append(outcome)

        self.trial_number += 1

    def calculate_p_explore(self):
        return EPSILON * (1 + self.surprises[-1] + np.mean(self.surprises)) / 3 if len(self.surprises) > 0 \
            else EPSILON / 3
