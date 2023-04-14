import functools

ALLOCATION_DICT = {"NONE": (0, 0), "LEFT": (1, 0), "RIGHT": (0, 1), "BOTH": (1, 1)}


@functools.cache
def assignments_per_turn(t, rr, lr, choices_per_reward):
    """
    :param t: Number of trials left
    :param rr: rewards left to assign to right side
    :param lr: rewards left to assign to left side
    :return: total number of rewards schedules possible for current trial.
    """
    if t == 0:  # We have finished the last trial meaning this is a leaf within recursion tree
        return 1
    if rr == t and lr == t:  # If we must assign to both sides
        total = different_choices(t - 1, rr - 1, lr - 1, choices_per_reward, ALLOCATION_DICT["BOTH"])
    elif rr == t:  # If we must assign to right side
        total = different_choices(t - 1, rr - 1, lr - 1, choices_per_reward, ALLOCATION_DICT["BOTH"]) \
                + different_choices(t - 1, rr - 1, lr, choices_per_reward, ALLOCATION_DICT["RIGHT"])
    elif lr == t:  # If we must assign to left side
        total = different_choices(t - 1, rr - 1, lr - 1, choices_per_reward, ALLOCATION_DICT["BOTH"]) \
                + different_choices(t - 1, rr, lr - 1, choices_per_reward, ALLOCATION_DICT["LEFT"])
    else:  # We assign all 4 different assignments
        total = different_choices(t - 1, rr, lr, choices_per_reward, ALLOCATION_DICT["NONE"]) \
                + different_choices(t - 1, rr - 1, lr, choices_per_reward, ALLOCATION_DICT["RIGHT"]) \
                + different_choices(t - 1, rr, lr - 1, choices_per_reward, ALLOCATION_DICT["LEFT"]) \
                + different_choices(t - 1, rr - 1, lr - 1, choices_per_reward, ALLOCATION_DICT["BOTH"])
    return total


def different_choices(t, rr, lr, choices_per_reward, allocation):
    # For each assignment the Agent has two different choices which effect the dynamic scheduling and the agent choice
    # doesn't make a difference when static schedule
    if t == 1:
        return assignments_per_turn(t, rr, lr, choices_per_reward)
    return choices_per_reward * assignments_per_turn(t, rr, lr, choices_per_reward)


if __name__ == '__main__':
    N_TRIALS = 100
    N_REWARDS = 25
    print(f"Number of dynamic schedules: \n{assignments_per_turn(N_TRIALS, N_REWARDS, N_REWARDS, 2)}")
