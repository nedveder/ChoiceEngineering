import numpy as np

TAO = 0.29
EPSILON = 0.30
PHI = 0.71
K = 2
ALTERNATIVE_A = 1
ALTERNATIVE_B = 0


class CatieAgent:
    previous_choices = []
    alt_a_rewards = []
    alt_b_rewards = []
    trial_num = 0
    p_explore = 0
    k = 0
    ca_alt_1 = 0
    ca_alt_2 = 0
    reward_history_dict = dict()
    all_surprise = []

    def __init__(self):
        # List of previous choices, where 1's indicate a
        # choice of alternative A and 0's indicate choice of alternative B.
        self.previous_choices = []
        # List of observed rewards for Alternative A on each trial
        self.alt_a_rewards = []
        # List of observed rewards for Alternative B on each trial
        self.alt_b_rewards = []
        self.p_explore = 0
        self.ca_alt_1 = 0
        self.ca_alt_2 = 0
        self.all_surprise = []
        self.k = np.random.randint(0, K+1)
        self.choice_type_history = ""

    def set_reward(self, reward):
        self.alt_a_rewards.append(reward[0])
        self.alt_b_rewards.append(reward[1])

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
        # Calculate Contingent Average and p_explore parameters.
        self.ca_alt_1, self.ca_alt_2 = self.contingent_average()
        self.p_explore = self.calculate_p_explore()

        # INIT MODE
        # For the first two trials we sample both options in random order.
        if len(self.previous_choices) < 2:
            # Samples both alternatives with random order
            choice = self.random_choice() if len(self.previous_choices) == 0 else self.other_choice()
            self.choice_type_history += "0"

        # TREND MODE
        # If we chose the same alternative in the past two trials and received different outcomes then with probability
        # TAO we enter the TREND mode.
        elif self.previous_choices[len(self.previous_choices) - 1] == self.previous_choices[len(self.previous_choices) - 2] \
                and (self.alt_a_rewards[len(self.previous_choices) - 1] != self.alt_a_rewards[len(self.previous_choices) - 2]
                     or self.alt_b_rewards[len(self.previous_choices) - 1] != self.alt_b_rewards[len(self.previous_choices) - 2]) \
                and np.random.random() <= TAO:
            choice = self.trend()
            self.choice_type_history += "T"

        # EXPLORE MODE
        # Explore is entered with probability p_explore which is determined using calculate_p_explore method at
        # each trial. Explore mode means choosing at random from both alternatives.
        elif np.random.random() <= self.p_explore:
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
            if self.ca_alt_1 == self.ca_alt_2:
                choice = self.random_choice()
            else:
                choice = ALTERNATIVE_A if self.ca_alt_1 > self.ca_alt_2 else ALTERNATIVE_B
            self.choice_type_history += "A"

        self.previous_choices.append(choice)
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
        matching_choice_rewards = self.alt_a_rewards if self.previous_choices[-1] == 1 else self.alt_b_rewards
        if matching_choice_rewards[- 1] > matching_choice_rewards[- 2]:
            return self.previous_choices[-1]
        else:
            return self.other_choice()

    @staticmethod
    def overlapping_sublist(lst, sub_lst):
        """
        Using a sliding window across the list we get all occurrences of sub list in list
        :return: A list of indexes for the start of each time the sub_lst occurred.
        """
        return [i for i in range(len(lst) - len(sub_lst) + 1) if lst[i: i + len(sub_lst)] == sub_lst]

    def contingent_average(self):
        alt_a_contingency_rewards = []
        alt_b_contingency_rewards = []
        if len(self.previous_choices) < 2:
            return 0, 0
        if self.k == 0:
            k_contingencies = list(range(0, len(self.previous_choices) - 1))
        else:
            k_contingency = self.previous_choices[-self.k:]
            k_contingencies = self.overlapping_sublist(self.previous_choices, k_contingency)

        for trial in k_contingencies:
            # Checks whether this is the last k_contingency meaning there is no element afterwards,
            # so we can't check if we received a reward , in such a case we skip.
            if trial + self.k >= len(self.previous_choices):
                continue
            # If the next choice was Alternative A
            elif self.previous_choices[trial + self.k] == ALTERNATIVE_A:
                alt_a_contingency_rewards.append(self.alt_a_rewards[trial + self.k])
            # otherwise the next choice was Alternative B
            elif self.previous_choices[trial + self.k] == ALTERNATIVE_B:
                alt_b_contingency_rewards.append(self.alt_b_rewards[trial + self.k])

        if len(alt_a_contingency_rewards) == 0:
            alt_a_contingency_rewards = self.no_k_contingency(ALTERNATIVE_A)
        if len(alt_b_contingency_rewards) == 0:
            alt_b_contingency_rewards = self.no_k_contingency(ALTERNATIVE_B)

        return np.mean(alt_a_contingency_rewards), np.mean(alt_b_contingency_rewards)

    def no_k_contingency(self, alternative):
        """
        If for alternative 𝑖 there are no past k-contingencies matching the
        most recent contingency, the value of a different random k-
        contingency replaces the contingent average for that
        alternative. A random k-contingency value is obtained by
        considering the set of all reward sequences of length 𝑘 which
        were followed by choice of 𝑖, randomly choosing one of them
        with a uniform probability and calculating its contingent
        average value. Note that choosing uniformly from past trials
        implies that recurring contingencies have higher probability of
        being chosen. If there are no contingencies of length 𝑘 that are
        followed by a choice of 𝑖, a smaller k (k:=k-1) is chosen
        iteratively until at least one contingency exists.
        :param alternative: 0 or 1
        :return: list of current alternative contingencies
        """
        cur_k = self.k
        alt_contingency_rewards = []
        alt_rewards = self.alt_a_rewards if self.previous_choices[-1] == 1 else self.alt_b_rewards
        while len(alt_contingency_rewards) == 0:
            if cur_k == 0:
                optional_k_contingencies = list(range(len(self.previous_choices)))
            else:
                optional_k_contingencies = self.overlapping_sublist(self.previous_choices[:-cur_k], [alternative])

            if optional_k_contingencies:
                index = np.random.choice(optional_k_contingencies)
                k_contingencies = self.overlapping_sublist(self.previous_choices[cur_k:],
                                                           self.previous_choices[index:index + cur_k] + [alternative])
                for trial in k_contingencies:
                    alt_contingency_rewards.append(alt_rewards[trial + cur_k])
            cur_k -= 1
        return alt_contingency_rewards

    def calculate_p_explore(self):
        if len(self.previous_choices) == 0:
            return 0
        cur_alt_rewards = self.alt_a_rewards if self.previous_choices[-1] else self.alt_b_rewards
        obs_sd = np.std(cur_alt_rewards)
        surprise_t = 0
        if obs_sd > 0:
            x = abs((self.ca_alt_1 if self.previous_choices[-1] else self.ca_alt_2) - cur_alt_rewards[-1])
            surprise_t = x / (obs_sd + x)
        self.all_surprise.append(surprise_t)
        return EPSILON * (1 + surprise_t + np.mean(self.all_surprise)) / 3
