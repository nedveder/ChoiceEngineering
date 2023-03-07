import numpy as np

TAO = 0.29
EPSILON = 0.30
PHI = 0.71
K = 2
OPTIONS = ["A", "B"]
ALTERNATIVE_A = 1
ALTERNATIVE_B = 0
REWARD = 1
NO_REWARD = 0


class CatieModel:
    choice_history = ""  # String containing choices "A" for alternative 1 or "B" for alternative 2
    reward_history = ""  # String containing rewards observed 1 if seen else 0
    choices_allocations = []
    trial_num = 0
    p_explore = 0
    k = 0
    ca_alt_1 = 0
    ca_alt_2 = 0
    reward_history_dict = dict()
    all_surprise = []

    def __init__(self):
        self.choice_history = ""  # String containing choices "A" for alternative 1 or "B" for alternative 2
        self.reward_history = ""  # String containing rewards observed 1 if seen else 0
        self.choices_allocations = []
        self.trial_num = 0
        self.p_explore = 0
        self.ca_alt_1 = 0
        self.ca_alt_2 = 0
        self.all_surprise = []
        self.reward_history_dict = {"A": [], "B": []}
        self.k = np.random.randint(0, K + 1)
        self.choice_type_history = ""

    def choose(self, reward_func):
        choice = ""

        # INIT
        if self.trial_num < 2:
            # Samples both alternatives with random order
            choice = self.random_choice() if self.trial_num == 0 else self.other_choice()
            self.choice_history += choice
            reward = reward_func(self.trial_num, choice)
            self.reward_history_dict[choice].append(reward)
            self.reward_history += str(reward)
            self.trial_num += 1
            self.choice_type_history += "0"
            self.calculate_p_explore()
            return choice

        self.calculate_p_explore()
        self.ca_alt_1, self.ca_alt_2 = self.contingent_average()

        # TREND
        if self.choice_history[self.trial_num - 1] == self.choice_history[self.trial_num - 2] \
                and self.reward_history[self.trial_num - 1] != self.reward_history[self.trial_num - 2] \
                and np.random.random() <= TAO:
            choice = self.trend()
            self.choice_type_history += "T"

        # EXPLORE
        elif np.random.random() <= self.p_explore:
            choice = self.explore()
            self.choice_type_history += "E"

        # INERTIA
        elif np.random.random() <= PHI:
            choice = self.inertia()
            self.choice_type_history += "I"

        # CONTINGENT AVERAGE
        else:
            if self.ca_alt_1 == self.ca_alt_2:
                choice = self.random_choice()
            else:
                choice = OPTIONS[0] if self.ca_alt_1 > self.ca_alt_2 else OPTIONS[1]
            self.choice_type_history += "A"

        self.choice_history += choice
        reward = reward_func(self.trial_num, choice)
        self.reward_history_dict[choice].append(reward)
        self.reward_history += str(reward)
        self.trial_num += 1
        return choice

    @staticmethod
    def random_choice():
        return OPTIONS[np.random.randint(0, 2)]

    def other_choice(self):
        return OPTIONS[(OPTIONS.index(self.choice_history[self.trial_num - 1]) + 1) % 2]

    def trend(self):
        return self.choice_history[self.trial_num - 1] \
            if self.reward_history[self.trial_num - 1] > self.reward_history[self.trial_num - 2] \
            else self.other_choice()

    def explore(self):
        return self.random_choice()

    def inertia(self):
        return self.choice_history[self.trial_num - 1]

    @staticmethod
    def overlapping_substring(string, pattern):
        return [i for i in range(len(string) - len(pattern) + 1) if string[i: i + len(pattern)] == pattern]

    def contingent_average(self):
        sum_alt_1, count_alt_1, sum_alt_2, count_alt_2 = 0, 0, 0, 0

        if self.k == 0:
            k_contingencies = list(range(0, self.trial_num - 1))
        else:
            k_contingencies = self.overlapping_substring(self.choice_history, self.choice_history[-self.k:])

        for trial in k_contingencies:
            if trial + self.k == self.trial_num:
                break
            if self.choice_history[trial + self.k] == OPTIONS[0]:
                sum_alt_1 += int(self.reward_history[trial + self.k])
                count_alt_1 += 1
            else:
                sum_alt_2 += int(self.reward_history[trial + self.k])
                count_alt_2 += 1

        if count_alt_1 == 0:
            count_alt_1, sum_alt_1 = self.no_k_contingency(OPTIONS[0])
        if count_alt_2 == 0:
            count_alt_2, sum_alt_2 = self.no_k_contingency(OPTIONS[1])
        return (sum_alt_1 / count_alt_1), (sum_alt_2 / count_alt_2)

    def no_k_contingency(self, alternative):
        cur_k = self.k
        sum_alt, count_alt = 0, 0
        while count_alt == 0:
            if cur_k == 0:
                optional_k_contingencies = list(range(0, self.trial_num - 1))
            else:
                optional_k_contingencies = self.overlapping_substring(self.choice_history[:-cur_k], alternative)

            if optional_k_contingencies:
                index = np.random.choice(optional_k_contingencies)
                k_contingencies = self.overlapping_substring(self.choice_history[cur_k:],
                                                             self.choice_history[index:index + cur_k] + alternative)
                for trial in k_contingencies:
                    sum_alt += int(self.reward_history[trial + cur_k - 1])
                    count_alt += 1
            cur_k -= 1
        return count_alt, sum_alt

    def calculate_p_explore(self):
        obs_sd = np.std(self.reward_history_dict[self.choice_history[-1]])
        surprise_t = 0
        if obs_sd > 0:
            x = abs((self.ca_alt_1 if self.choice_history[-1] == OPTIONS[0] else self.ca_alt_2)
                    - int(self.reward_history[-1]))
            surprise_t = x / (obs_sd + x)
        self.all_surprise.append(surprise_t)
        self.p_explore = EPSILON * (1 + surprise_t + np.mean(self.all_surprise)) / 3
