import CATIE_model
import matplotlib.pyplot as plt
import numpy as np

ITERATIONS = 100
TRIALS = 100
BASIC_SCHEDULE = "1111111111000000000000000000000000000000000000000000000000000000000000000000000000000000002222222222"


def reward_func(trial_num, choice):
    if BASIC_SCHEDULE[trial_num] == "0":
        return 0
    if BASIC_SCHEDULE[trial_num] == "2" and choice == CATIE_model.OPTIONS[0]:
        return 0
    if BASIC_SCHEDULE[trial_num] == "1" and choice == CATIE_model.OPTIONS[1]:
        return 0
    return 1


def main():
    iterations = []
    for _ in range(ITERATIONS):
        ca_model = CATIE_model.CatieModel()
        for __ in range(TRIALS):
            ca_model.choose(reward_func)
        iterations.append(ca_model.choice_history.count("A") / TRIALS)
        print(ca_model.reward_history)
        print(ca_model.choice_history)
        print(ca_model.choice_type_history)

    n, bins, patches = plt.hist(iterations, 50, density=True, facecolor='g', alpha=0.75)
    plt.axvline(np.array(iterations).mean(), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(np.array(iterations).mean() * 1.1, max_ylim * 0.9, 'Mean: {:.3f}'.format(np.array(iterations).mean()))
    plt.show()


if __name__ == '__main__':
    np.random.seed(1)
    main()
