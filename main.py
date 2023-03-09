import statistics

import CatieAgent
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

CHOICE_BIASED = 1
CHOICE_ANTI_BIASED = 0


def sequence_catie_score(reward_schedule, repetitions=100, plot_distribution=False, plot_sequence=False):
    schedule_target, schedule_anti_target = reward_schedule[0], reward_schedule[1]
    biases = []
    for _ in tqdm(range(repetitions)):
        catie_agent = CatieAgent.CatieAgent(len(schedule_target))
        choices = []
        for reward_target, reward_anti_target in zip(schedule_target, schedule_anti_target):
            choice = catie_agent.choose()
            outcome = reward_anti_target, reward_target
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
        plt.hist(biases, alpha=0.5, density=True)
        plt.ylabel('Probability')
        plt.xlabel('Bias')
        plt.axvline(statistics.mean(biases), color='k', linestyle='dashed', linewidth=1)
        min_ylim, max_ylim = plt.ylim()
        plt.text(statistics.mean(biases) * 1.1, max_ylim * 0.9, 'Mean: {:.2f}'.format(statistics.mean(biases)))
    return biases, statistics.mean(biases)


def valid_one_side_allocation(shuffle=False):
    valid_seq = np.concatenate((np.zeros(75), np.ones(25)))
    if shuffle:
        np.random.shuffle(valid_seq)
    return valid_seq


def random_valid_sequence():
    return valid_one_side_allocation(True), valid_one_side_allocation(True)


def catie_naive_optimization_seq():
    target_side = np.concatenate((np.ones(25), np.zeros(75)))
    anti_target_side = np.concatenate((np.zeros(75), np.ones(25)))
    return target_side, anti_target_side


def main():
    random_test()
    qlearning_test()
    catie_test()
    opt_catie_test()
    comp_winner_test()
    plt.show()


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
    mean_bias_optimized = sequence_catie_score(winner_schedule, 1000, True, True)[1]


def opt_catie_test():
    optimized_switched = np.array(
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1.,
         0., 1., 1., 1.]), np.array(
        [1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
         0., 0., 0., 0., ])
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
    mean_bias_optimized = sequence_catie_score(optimized, 1000, True)[1]
    mean_bias_optimized_switched = sequence_catie_score(optimized_switched, 1000, True)[1]


def catie_test():
    mean_bias = sequence_catie_score(catie_naive_optimization_seq(), 1000, True)[1]


def qlearning_test():
    ql_fit_hom = np.array(
        [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
         1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0.,
         1., 0., 1., 1., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), np.array(
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0.,
         0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    mean_bias_ql_fit_hom = sequence_catie_score(ql_fit_hom, 1000, True, True)[1]


NUMBER_OF_RANDOM_SEQ = 100
REPETITIONS_PER_SEQ = 10


def random_test():
    random_seq_mean_biases = [sequence_catie_score(random_valid_sequence(), REPETITIONS_PER_SEQ)[1] for i in
                              range(NUMBER_OF_RANDOM_SEQ)]
    plt.hist(random_seq_mean_biases, alpha=0.5, density=True)
    plt.ylabel('Probability')
    plt.xlabel('Bias')
    plt.axvline(statistics.mean(random_seq_mean_biases), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(statistics.mean(random_seq_mean_biases) * 1.1, max_ylim * 0.9,
             'Mean: {:.2f}'.format(statistics.mean(random_seq_mean_biases)))


if __name__ == '__main__':
    np.random.seed(1)
    main()
