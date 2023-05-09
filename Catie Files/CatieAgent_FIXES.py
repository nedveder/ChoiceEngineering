import random
import statistics

import numpy as np
import tqdm
from matplotlib import pyplot as plt
from CatieAgentC import CatieAgent

def sequence_catie_score(reward_schedule, repetitions=100, plot_distribution=False):
    """Calculate the sequence Catie score for a given reward schedule.

    Args:
        reward_schedule (tuple): A tuple containing the target and anti-target reward schedules.
        repetitions (int, optional): The number of repetitions to perform. Defaults to 100.
        plot_distribution (bool, optional): Whether to plot the distribution of biases. Defaults to False.
        plot_sequence (bool, optional): Whether to plot the sequence of rewards. Defaults to False.

    Returns:
        tuple: A tuple containing the biases and the mean bias.
    """
    schedule_target, schedule_anti_target = reward_schedule
    biases = []

    for _ in tqdm.trange(repetitions):
        catie_agent = CatieAgent()
        choices = []

        for t, (reward_target, reward_anti_target) in enumerate(zip(schedule_target, schedule_anti_target)):
            choice = catie_agent.choose()
            outcome = reward_target, reward_anti_target
            catie_agent.receive_outcome(choice, outcome)

            choices.append(choice)

        biases.append(sum(choices))

    if plot_distribution:
        plot_bias_distribution(biases)

    return biases, statistics.mean(biases)



def plot_bias_distribution(biases):
    plt.figure()
    plt.hist(biases, color='royalblue', alpha=0.5, density=True)
    plt.ylabel('Probability')
    plt.xlabel('Bias')
    plt.axvline(statistics.mean(biases), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(plt.xlim()[0] * 1.1, plt.ylim()[1] * 0.95, 'Mean: {:.3f}'.format(statistics.mean(biases)))


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
    plt.title("CATIE Static Naive Nadav's implementation")
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
    plt.title("CATIE Static Winner Nadav's implementation")
    plt.text(plt.xlim()[0] * 1.1, plt.ylim()[1] * 0.9,
             f'error: +/-{statistics.pstdev(mean_bias_optimized[0]) / np.sqrt(N):.3f}%')
    plt.text(plt.xlim()[0] * 1.1, plt.ylim()[1] * 0.85, f'N: {N}')


if __name__ == '__main__':
    np.random.seed(1)
    random.seed(1)
    test_catie_opt()
    comp_winner_test()
    plt.show()
