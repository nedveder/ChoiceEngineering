import numpy as np

TAO = 0.29
EPSILON = 0.30
PHI = 0.71
K = 2
ALTERNATIVE_A = 1
ALTERNATIVE_B = 0


class CatieAgent:

    def __init__(self, number_of_trials, tao=TAO, epsilon=EPSILON, phi=PHI, k=K):
        # Contingent Average value for each choice at each trial
        self.number_of_trials = number_of_trials
        self.alt_b_contingent_average = 0
        self.alt_a_contingent_average = 0
        # Dictionaries containing all k-contingencies so far
        self.alt_a_contingencies = dict()
        self.alt_b_contingencies = dict()
        # Array of previous choices, where 1's indicate a
        # choice of alternative A and 0's indicate choice of alternative B.
        self.previous_choices = np.zeros(number_of_trials, dtype=np.int8)
        # Array of all outcomes
        self.outcomes = np.zeros(number_of_trials, dtype=np.int8)
        # Arrays of outcomes for each Alternative on each trial that it was chosen
        self.alt_a_outcomes = np.zeros(number_of_trials, dtype=np.int8)
        self.alt_b_outcomes = np.zeros(number_of_trials, dtype=np.int8)
        # Indexes that count how many times each choice was made.
        self.choice_indexes = {ALTERNATIVE_A: 0, ALTERNATIVE_B: 0}
        # Index of current trial
        self.trial_number = 0
        # Array of surprise_t values at each trial
        self.surprises = np.zeros(number_of_trials, dtype=np.float64)
        # All of CATIE agent parameters
        self.tao = tao
        self.epsilon = epsilon
        self.phi = phi
        self.k = np.random.randint(0, k + 1)

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

        # Compute current contingencies for each alternative.
        self.alt_a_contingent_average = self.contingent_average(self.k, ALTERNATIVE_A)
        self.alt_b_contingent_average = self.contingent_average(self.k, ALTERNATIVE_B)

        # INIT MODE
        # For the first two trials we sample both options in random order.
        if self.trial_number < 2:
            # Samples both alternatives with random order
            choice = self.random_choice() if self.trial_number == 0 else self.other_choice()

        # TREND MODE
        # If we chose the same alternative in the past two trials and received different outcomes then with probability
        # TAO we enter the TREND mode.
        elif self.previous_choices[self.trial_number - 1] == self.previous_choices[self.trial_number - 2] \
                and self.outcomes[self.trial_number - 1] != self.outcomes[self.trial_number - 2] \
                and np.random.random() <= self.tao:
            choice = self.trend()

        # EXPLORE MODE
        # Explore is entered with probability p_explore which is determined using calculate_p_explore method at
        # each trial. Explore mode means choosing at random from both alternatives.
        elif np.random.random() <= self.calculate_p_explore():
            choice = self.random_choice()

        # INERTIA MODE
        # Choose the previous choice again.
        elif np.random.random() <= self.phi:
            choice = self.previous_choices[self.trial_number - 1]

        # CONTINGENT AVERAGE MODE
        # In the contingent average (CA) mode, the agent chooses the alternative associated with the higher k-CA,
        # defined as the average payoff observed in all previous trials which followed the same sequence of k outcomes
        # See contingent_average method for further information.
        else:
            if self.alt_a_contingent_average == self.alt_b_contingent_average:
                choice = self.random_choice()
            else:
                # Alternative A if self.alt_a_contingent_average > self.alt_b_contingent_average else Alternative B
                choice = int(self.alt_a_contingent_average > self.alt_b_contingent_average)

        return choice

    @staticmethod
    def random_choice():
        """
        :return: A random choice between the two alternatives.
        """
        return np.random.randint(0, 2)  # random choice between Alternative A or Alternative B

    def other_choice(self):
        """
        :return: Given the previous choice return the other alternative.
        """
        return (self.previous_choices[self.trial_number - 1] + 1) % 2

    def trend(self):
        """
        In this mode the agent chooses the “positive trend” meaning if the outcome of the previous choice was
        bigger then the one preceding it, we choose it ̧otherwise we switch choices.
        :return: The choice corresponding with TREND mode.
        """
        return self.previous_choices[self.trial_number - 1] if self.outcomes[self.trial_number - 1] > self.outcomes[
            self.trial_number - 2] else self.other_choice()

    def contingent_average(self, k, choice):
        """
        The contingent average of the input alternative.
        """
        choice_outcomes = self.alt_a_outcomes if choice else self.alt_b_outcomes
        if k == 0:
            return np.mean(choice_outcomes[:self.choice_indexes[choice]]) if self.choice_indexes[choice] else 0
        # Get last k choices and outcomes
        current_contingency = tuple(zip(self.previous_choices[self.trial_number - self.k:self.trial_number],
                                        self.outcomes[self.trial_number - self.k:]))
        # Get appropriate contingency dictionary
        choice_contingencies = self.alt_a_contingencies if choice else self.alt_b_contingencies
        # If contingency already exists in dictionary then return it
        if current_contingency in choice_contingencies:
            index, contingency_values = choice_contingencies[current_contingency]
            return np.mean(contingency_values[:index])
        # Current contingency does not exist
        if not choice_contingencies:  # If there are no k-length contingencies simply consider the mean
            return np.mean(choice_outcomes[:self.choice_indexes[choice]]) if self.choice_indexes[choice] else 0
        else:
            # The current contingency does not appear, but other contingencies do. In such case the user gets
            # "confused" and chooses one of the other contingencies at random. This scenario return all the possible
            # outcomes of the confusion (which are by definition chosen with uniform probability). Namely, the average
            # of each existing contingent average outcome
            return np.mean([con_array[1] for con_array in choice_contingencies.values()])

    def receive_outcome(self, choice, outcome):
        choice_outcomes = self.alt_a_outcomes if choice else self.alt_b_outcomes
        choice_outcomes[self.choice_indexes[choice]] = outcome[choice]
        self.choice_indexes[choice] += 1
        obs_sd = 0 if self.trial_number < 2 else np.std(choice_outcomes[:self.choice_indexes[choice]])
        if obs_sd:  # If standard deviation in positive
            # Expected reward for choice on current trial
            exp_t_i = self.alt_a_contingent_average if choice else self.alt_b_contingent_average
            expected_actual_reward_diff = abs(exp_t_i - outcome[choice])
            surprise_t = expected_actual_reward_diff / (obs_sd + expected_actual_reward_diff)
        else:
            surprise_t = 0

        # Update history
        self.surprises[self.trial_number] = surprise_t
        self.previous_choices[self.trial_number] = choice
        self.outcomes[self.trial_number] = outcome[choice]
        # Update contingent dictionary
        if 0 < self.k < self.trial_number:
            # Get previous k contingencies (before current trial)
            current_contingency = tuple(zip(self.previous_choices[self.trial_number - self.k:self.trial_number],
                                            self.outcomes[self.trial_number - self.k:self.trial_number]))
            choice_contingencies = self.alt_a_contingencies if choice else self.alt_b_contingencies
            if current_contingency not in choice_contingencies:
                choice_contingencies[current_contingency] = [0, np.zeros(self.number_of_trials)]
            choice_contingencies[current_contingency][1][choice_contingencies[current_contingency][0]] = outcome[choice]
            choice_contingencies[current_contingency][0] += 1

        self.trial_number += 1

    def calculate_p_explore(self):
        return self.epsilon / 3 * ((1 + self.surprises[self.trial_number - 1] + np.mean(
            self.surprises[:self.trial_number])) if self.trial_number else 1)
