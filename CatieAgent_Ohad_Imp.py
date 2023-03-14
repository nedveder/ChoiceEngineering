import numpy as np
import random
import statistics

import tqdm
from matplotlib import pyplot as plt

########################################################################################################################
# Constants
########################################################################################################################

PHI = 0.71
TAU = 0.29
EPSILON = 0.3

CHOICE_BIASED = 1
CHOICE_ANTI_BIASED = 0


########################################################################################################################
# Class helper functions
########################################################################################################################


def choose_randomly():
    """ Return either of the alternatives with 50% chance """
    if np.random.random() < 0.5:
        choice = CHOICE_BIASED
    else:
        choice = CHOICE_ANTI_BIASED
    return choice


def other_alternative(a):
    """
  Given alternative a return the other alternative:
    CHOICE_BIASED --> CHOICE_ANTI_BIASED
    CHOICE_ANTI_BIASED --> CHOICE_BIASED
  """
    if a is CHOICE_BIASED:
        return CHOICE_ANTI_BIASED
    if a is CHOICE_ANTI_BIASED:
        return CHOICE_BIASED


class Trend:
    """
    Assign a numerical representation to each of the trend types
    """
    POSITIVE = 1
    NON_POSITIVE = -1
    INVALID = 0


class History:
    """
    Constants to describe all possible outcomes of a trial, used to calculate contingencies
    """
    BIASED_REWARD = 'BIASED_REWARD'
    BIASED_NO_REWARD = 'BIASED_NO_REWARD'
    ANTI_BIASED_REWARD = 'ANTI_BIASED_REWARD'
    ANTI_BIASED_NO_REWARD = 'ANTI_BIASED_NO_REWARD'


def trial_to_history(choice, outcome):
    if choice == CHOICE_BIASED and outcome == 1:
        return History.BIASED_REWARD
    elif choice == CHOICE_BIASED and outcome == 0:
        return History.BIASED_NO_REWARD
    elif choice == CHOICE_ANTI_BIASED and outcome == 1:
        return History.ANTI_BIASED_REWARD
    elif choice == CHOICE_ANTI_BIASED and outcome == 0:
        return History.ANTI_BIASED_NO_REWARD
    else:
        raise Exception('Invalid choice and outcome combination')


def biased_side_p_for_ca(ca_biased, ca_anti_biased):
    """
    Return the probability of choosing the biased side given the contingent average of the biased and anti
        biased side. The contingent average is considered "exploitation" mode, thus the probability of choosing
        the biased side is 1 if ca_biased>ca_anti_biased, and 0 if ca_biased<ca_anti_biased. If the averages are
        equal, the two alternatives are chosen with equal probability.
    :param ca_biased: The contingent average of the biased side.
    :param ca_anti_biased: The contingent average of the anti-biased side.
    :return: The probability of choosing the biased side
    """
    if ca_biased > ca_anti_biased:
        return 1
    elif ca_biased < ca_anti_biased:
        return 0
    elif ca_biased == ca_anti_biased:
        return 0.5
    else:
        raise Exception('Invalid value comparison. Biased: ' + str(ca_biased) +
                        ',  anti biased:  ' + str(ca_anti_biased))


########################################################################################################################
# Class implementation
########################################################################################################################


class CatieAgent:
    """
    One difference in the current implementation from the published one is the handling of choice probabilities while
        the agent has not sampled both alternatives yet. In the original implementation, the agent is enforced to choose
        the two alternatives in the first two trials. In behavior, participants do not have such a constraint. This is a
        relatively rare case since most participants do sample the two alternatives fairly quickly. Still, it has to be
        covered. To handle this scenario, this implementation assumes that in the "exploit" mode (contingent average)
        the agent continues choosing the chosen alternative. Such implementation choice is consistent with the
        proposition that the reason for not choosing the chosen alternative is the agent's believe that the chosen
        alternative is better and hence "exploitation" implies continue choosing it.
    """

    def __init__(self, K=2, k=None, is_model_for_fitting=False):
        self.tau = TAU
        self.phi = PHI
        self.epsilon = EPSILON
        self.K = K
        self.k = k if k is not None else random.randint(0, K)
        self.trial_number = 0
        self.choices = []
        self.surprises = []
        self.outcomes = []
        self.outcomes_biased = []
        self.outcomes_anti_biased = []
        self.history = []
        self.biased_contingencies = dict()
        self.anti_biased_contingencies = dict()

        self.is_model_for_fitting = is_model_for_fitting
        self.actions_likelihood = 1  # Normalized prior, multiplied by action probability after each action
        self.target_side_choice_probability = None  # Maintain the current-trial's probability of choice in the target
        # side

    def get_all_agents(self):
        """
        Implemented for compatability with classes that hold several independent decision making agents. For this class,
        this is equivalent to returning self (which is indeed all the internally maintained agents).
        :return:
        """
        return [self]

    def get_p_explore(self):
        if len(self.surprises) == 0:
            p_explore = self.epsilon * 1 / 3
        else:
            p_explore = self.epsilon * (1 + self.surprises[-1] + np.mean(self.surprises)) / 3
        return p_explore

    def get_trend(self):
        """
        Probability for entering Explore mode
        """
        if (len(self.choices) < 2 or
                self.choices[-1] != self.choices[-2]):
            trend = Trend().INVALID
        elif self.outcomes[-1] > self.outcomes[-2]:
            trend = Trend().POSITIVE
        else:
            trend = Trend().NON_POSITIVE
        return trend

    def choose(self):
        """
        Return a single choice based on current state using the four decision
        modes.
        """
        biased_side_probability = self.biased_side_choice_probability()
        choice = CHOICE_BIASED if random.random() < biased_side_probability else CHOICE_ANTI_BIASED
        # global biased_side_probabilities
        # biased_side_probabilities.append(biased_side_probability)
        return choice

    def biased_side_choice_probability(self):
        """
        Return the probability of choosing the biased alternative.
        In the original implementation, it is assumed that both alternatives are sampled within the first two trials.
            That makes the probability of choosing the same alternative twice in the first two trials 0. Since this
            is a possible scenario, the implementation here doesn't enforce any choice in the second trial.
        """
        if self.trial_number == 0:  # First choice is random
            p_biased = 0.5
        elif self.trial_number == 1 and not self.is_model_for_fitting:  # Second choice is in the alternative not chosen
            # in previous trial; but this isn't enforced when fitting the model to behavior (becuase humans do not
            # have to comply with this rule)
            p_biased = 1 if self.choices[-1] == CHOICE_ANTI_BIASED else 0
        else:
            p_biased = 0
            # Trend/Heuristic mode
            # if the same alternative was chosen in the last two trials, and it had different outcomes,
            is_test_trend = (self.trial_number > 1 and
                             self.choices[-1] == self.choices[-2]) and \
                            (self.outcomes[-1] != self.outcomes[-2])
            # If conditions apply, enter heuristic mode with probability tau
            if is_test_trend:
                probability_of_entering_trend_mode = self.tau
                # If there is a positive trend, continue choosing the same alternative, if not switch
                if (self.outcomes[-1] > self.outcomes[-2]) and (self.choices[-1] == CHOICE_BIASED) or \
                        (self.outcomes[-1] <= self.outcomes[-2]) and (self.choices[-1] == CHOICE_ANTI_BIASED):
                    p_biased += probability_of_entering_trend_mode

            # Explore mode
            # with p_explore probability, in which choose randomly
            probability_of_trying_explore_mode = 1 if not is_test_trend else (1 - self.tau)
            p_explore = self.get_p_explore()
            probability_of_entering_explore_mode = probability_of_trying_explore_mode * p_explore
            p_biased += probability_of_entering_explore_mode * 0.5

            # Inertia mode
            # With probability phi, choose the same
            probability_of_trying_inertia_mode = probability_of_trying_explore_mode * (
                    1 - probability_of_entering_explore_mode)
            probability_of_entering_inertia_mode = probability_of_trying_inertia_mode * self.phi
            p_biased += probability_of_entering_inertia_mode if self.choices[-1] == CHOICE_BIASED else 0

            # Contingent average mode
            probability_of_entering_ca_mode = probability_of_trying_inertia_mode * (1 - self.phi)
            if not self.outcomes_biased or not self.outcomes_anti_biased:  # If one of the alternatives was not sampled yet
                # Then "exploit" means continue choosing the already chosen alternative
                contingency_mode_biased_probability = probability_of_entering_ca_mode if self.outcomes_biased else 1-probability_of_entering_ca_mode
            else:
                if self.k > 0:
                    all_contingencies_probabilities = [biased_side_p_for_ca(ca_biased, ca_anti_biased)
                                                       for ca_biased in self.__contingent_average(self.k, CHOICE_BIASED)
                                                       for ca_anti_biased in self.__contingent_average(self.k, CHOICE_ANTI_BIASED)]
                    contingency_mode_biased_probability = probability_of_entering_ca_mode * np.mean(all_contingencies_probabilities)
                else:
                    target_mean, anti_target_mean = np.mean(self.outcomes_biased), np.mean(self.outcomes_anti_biased)
                    if target_mean == anti_target_mean:  # If the two alternative means are equal, choose randomly
                        contingency_mode_biased_probability = probability_of_entering_ca_mode * 0.5
                    else:  # Choose, deterministically, the alternative with the greater mean outcome
                        contingency_mode_biased_probability = probability_of_entering_ca_mode if target_mean > anti_target_mean else 0
            p_biased += contingency_mode_biased_probability
        self.target_side_choice_probability = p_biased
        return p_biased

    def __contingent_average(self, k, choice):
        """
        The contingent average of the input alternative.
        """
        if k == 0:
            choice_outcomes = self.outcomes_biased if choice == CHOICE_BIASED else self.outcomes_anti_biased
            return [np.mean(choice_outcomes)]
        else:
            current_contingency = tuple(self.history[-self.k:])  # Get last k choices and outcomes
            choice_contingencies = self.biased_contingencies if choice == CHOICE_BIASED else self.anti_biased_contingencies
            if current_contingency in choice_contingencies:
                return [np.mean(choice_contingencies[current_contingency])]
            else:  # Current contingency does not exist
                if not choice_contingencies:  # If there are no k-length contingencies simply consider the mean
                    choice_outcomes = self.outcomes_biased if choice == CHOICE_BIASED else self.outcomes_anti_biased
                    return [np.mean(choice_outcomes)]
                else:  # The current contingency does not appear, but other contingencies do. In such case the user gets
                    # "confused" and chooses one of the other contingencies at random. Since the current
                    # implementation ultimately computes probability, this scenario return all the possible outcomes
                    # of the confusion (which are by definition chosen with uniform probability). Namely, the average
                    # of each existing contingent average outcome
                    return [np.mean(confused_contingency_outcomes) for confused_contingency_outcomes in
                            choice_contingencies.values()]

    def receive_outcome(self, choice, outcome):
        """
        Updated internal state for receiving the input outcome as the result of
        choosing the input choice.
        """
        # Update surprise relative to the given outcome
        if choice == CHOICE_BIASED:
            choice_probability = self.target_side_choice_probability
            self.outcomes_biased.append(outcome)
            obs_sd = 0 if len(self.outcomes_biased) < 2 else np.std(self.outcomes_biased)  # stddv is only defined for 2 or more numbers
        elif choice == CHOICE_ANTI_BIASED:
            choice_probability = 1 - self.target_side_choice_probability
            self.outcomes_anti_biased.append(outcome)
            obs_sd = 0 if len(self.outcomes_anti_biased) < 2 else np.std(self.outcomes_anti_biased)
        self.actions_likelihood *= choice_probability
        if obs_sd > 0:
            # exp_t_i is the expected value the user excepts to get at trial t after choosing i (the variable choice
            # here). In the original implementation, this should be a single number (the contingent average). In the
            # current implementation, this is indeed often a single number (in which case, the mean calculation below
            # is redundant). However, if the current contingency does not exist, then the user is "confused" and picks
            # a different contingency at random (if such exist, see the __contingent_average implementation). Since this
            # implementations aims to generate choice probability, in such scenarios, rather than choosing one
            # contingency at random, the current implementation considers the expected outcome. That is, the average of
            # all potential confusions, namely the average of all existing contingencies.
            exp_t_i = np.mean(self.__contingent_average(self.k, choice))
            expected_actual_reward_diff = abs(exp_t_i - outcome)
            surprise_t = expected_actual_reward_diff / (obs_sd + expected_actual_reward_diff)
        else:
            surprise_t = 0
        self.surprises.append(surprise_t)

        # Update self with the current outcome
        self.choices.append(choice)
        self.outcomes.append(outcome)
        self.history.append(trial_to_history(choice, outcome))

        # Update contingent dictionary
        if 0 < self.k < len(self.choices):
            current_contingency = tuple(
                self.history[-self.k - 1:-1])  # Get previous k contingencies (before current trial)
            choice_contingencies = self.biased_contingencies if choice == CHOICE_BIASED else self.anti_biased_contingencies
            if current_contingency not in choice_contingencies:
                choice_contingencies[current_contingency] = []
            choice_contingencies[current_contingency].append(outcome)

        self.trial_number += 1


########################################################################################################################
# CATIE sequence score
########################################################################################################################


def sequence_catie_score(reward_schedule,
                         repetitions=100, plot_distribution=False,
                         plot_sequence=False):
    schedule_target, schedule_anti_target = reward_schedule[0], reward_schedule[1]
    biases = []
    for i in tqdm.trange(repetitions):
        catie_agent = CatieAgent()
        choices = []
        for t, (reward_targe, reward_anti_target) in enumerate(zip(
                schedule_target, schedule_anti_target)):
            choice = catie_agent.choose()
            outcome = reward_targe if choice == CHOICE_BIASED else reward_anti_target
            catie_agent.receive_outcome(choice, outcome)
            choices.append(choice)
        biases.append(sum(choices))
    if plot_sequence:
        plt.plot([i + 1 for i in range(100) if reward_schedule[0][i]], 2 * np.ones(25), 'x')
        plt.plot([i + 1 for i in range(100) if reward_schedule[1][i]], np.ones(25), 'x')
        plt.ylim([0.5, 2.5])
        plt.xlabel('Trial number')
        plt.ylabel('Is reward')
        plt.yticks([1, 2], ['Anti side', 'Target side'])
    if plot_distribution:
        plt.figure()
        plt.hist(biases, color='powderblue', alpha=0.5, density=True)
        plt.ylabel('Probability')
        plt.xlabel('Bias')
        plt.axvline(statistics.mean(biases), color='k', linestyle='dashed', linewidth=1)
        min_ylim, max_ylim = plt.ylim()
        plt.text(plt.xlim()[0] * 1.1, plt.ylim()[1] * 0.95, 'Mean: {:.3f}'.format(statistics.mean(biases)))
    return biases, statistics.mean(biases)


N = 1000


def test_catie_opt():
    """
    Test the performance of the potimized sequence vs. the "naive optimal" (25 rewards at the beginning of the target
    and at the end of the anti target). As sanity check, test the bias distribution of sending the optimized sequence
    where the target is anti targer and vice versa (should be symmetric around 50).
    """
    optimized = np.array(
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., ]), np.array(
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1.])

    mean_bias_optimized = sequence_catie_score(optimized, N, True)
    plt.title("CATIE Static Naive Ohad's implementation(Post-fix)")
    plt.text(plt.xlim()[0] * 1.1, plt.ylim()[1] * 0.9,
             f'error: +/-{statistics.pstdev(mean_bias_optimized[0]) / N:.3f}%')
    plt.text(plt.xlim()[0] * 1.1, plt.ylim()[1] * 0.85, f'N: {N}')


def comp_winner_test():
    winner_schedule = (np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 0., 0., 0., 1.,
                                 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0.,
                                 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                       np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                                 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1.,
                                 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                                 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0.,
                                 1., 1., 0., 0., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 1.]))
    mean_bias_optimized = sequence_catie_score(winner_schedule, N, True)
    plt.title("CATIE Static Winner Ohad's implementation(Post-fix)")
    plt.text(plt.xlim()[0] * 1.1, plt.ylim()[1] * 0.9,
             f'error: +/-{statistics.pstdev(mean_bias_optimized[0]) / np.sqrt(N):.3f}%')
    plt.text(plt.xlim()[0] * 1.1, plt.ylim()[1] * 0.85, f'N: {N}')


if __name__ == '__main__':
    np.random.seed(1)
    random.seed(1)
    test_catie_opt()
    comp_winner_test()
    plt.show()
