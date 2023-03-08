import CATIE_model
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import Scheduler

ITERATIONS = 10000
TRIALS = 100
BEST_CATIE_SCHEDULE = "1111111111011200111112011312000012000120110000000020000000000000200000020020202002020220022022022222"


def reward_func(trial_num, schedule, choice):
    biased_alt, non_biased_alt = 0, 0
    if schedule[trial_num] == "0":
        return 0, 0
    if (schedule[trial_num] == "1" or schedule[trial_num] == "3") and choice == CATIE_model.ALTERNATIVE_A:
        biased_alt = 1
    if (schedule[trial_num] == "2" or schedule[trial_num] == "3") and choice == CATIE_model.ALTERNATIVE_B:
        non_biased_alt = 1
    return non_biased_alt, biased_alt


def main():
    iterations = []
    for _ in tqdm(range(ITERATIONS)):
        ca_model = CATIE_model.CatieAgent(TRIALS)
        target_alloc = []
        non_target_alloc = []
        for t in range(TRIALS):
            choice = ca_model.choose()
            # outcome = Scheduler.allocate(target_alloc, non_target_alloc, ca_model.previous_choices)
            outcome = reward_func(t, BEST_CATIE_SCHEDULE, choice)
            ca_model.receive_outcome(choice, outcome)
            # target_alloc.append(outcome[1])
            # non_target_alloc.append(outcome[0])
        iterations.append(ca_model.previous_choices.sum() / TRIALS)
        # print(ca_model.previous_choices)
        # iterations.append(ca_model.previous_choices.count(1) / TRIALS)

    plt.hist(iterations, 50, density=True, facecolor='g', alpha=0.75)
    plt.axvline(np.array(iterations).mean(), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(np.array(iterations).mean() * 1.1, max_ylim * 0.9, 'Mean: {:.3f}'.format(np.array(iterations).mean()))
    plt.show()


if __name__ == '__main__':
    np.random.seed(1)
    main()
