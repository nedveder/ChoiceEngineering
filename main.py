import CATIE_model
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import Scheduler

ITERATIONS = 1000
TRIALS = 100
BASIC_SCHEDULE = "1111111111111111111111111000000000000000000000000000000000000000000000000002222222222222222222222222"


def reward_func(trial_num, schedule, choice):
    biased_alt, non_biased_alt = 0, 0
    if schedule[trial_num] == "0":
        return 0, 0
    if (schedule[trial_num] == "1" or schedule[trial_num] == "3") and choice == CATIE_model.ALTERNATIVE_A:
        biased_alt = 1
    if (schedule[trial_num] == "2" or schedule[trial_num] == "3") and choice == CATIE_model.ALTERNATIVE_B:
        non_biased_alt = 1
    return biased_alt, non_biased_alt


def main():
    iterations = []
    for _ in tqdm(range(ITERATIONS)):
        ca_model = CATIE_model.CatieAgent()
        for i in range(TRIALS):
            choice = ca_model.choose()
            ca_model.set_reward(reward_func(i, BASIC_SCHEDULE, choice))
        iterations.append(ca_model.previous_choices.count(CATIE_model.ALTERNATIVE_A) / TRIALS)

    n, bins, patches = plt.hist(iterations, 50, density=True, facecolor='g', alpha=0.75)
    plt.axvline(np.array(iterations).mean(), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(np.array(iterations).mean() * 1.1, max_ylim * 0.9, 'Mean: {:.3f}'.format(np.array(iterations).mean()))
    plt.show()


if __name__ == '__main__':
    np.random.seed(1)
    main()
