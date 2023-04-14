import statistics
import numpy as np
import tqdm
import heapq
import pickle
import scipy
from scipy import stats
import matplotlib.pyplot as plt
from CatieAgent import CatieAgent

NUMBER_OF_TRIALS = 100
# Network constants
L0_SIZE = 9  # Number of input neurons (reflecting current state, don't change)
L_MIDDLE_SIZE = 100  # Number of neurons per each of the middle layers (arbitrary, can be changed)
N_MIDDLE_LAYERS = 4  # Arbitrary, can be changed
# Number of neurons in the output layer, corresponding to the number of possible decisions (don't change)
L_END_SIZE = 4

# The greater the number, the greater is the mutation rate of each generation
WEIGHTS_INITIAL_VALUE_NORMALIZATION = 1 / 10

INVALID_VALUE = -1  # Invalid / nonexistent
NO_PREVIOUS_CHOICE = 0


def sequence_catie_score(reward_schedule, repetitions=100, plot_distribution=False, plot_sequence=False):
    schedule_target, schedule_anti_target = reward_schedule[0], reward_schedule[1]
    biases = []
    for _ in tqdm.tqdm(range(repetitions)):
        catie_agent = CatieAgent(len(schedule_target))
        choices = []
        for reward_target, reward_anti_target in zip(schedule_target, schedule_anti_target):
            choice = catie_agent.choose()
            outcome = reward_target, reward_anti_target
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
    return biases, np.mean(biases)


def random_weights(n_in, n_out):
    return (np.random.rand(n_in, n_out) - 0.5) * WEIGHTS_INITIAL_VALUE_NORMALIZATION


def init_nn(n_middle_layers=N_MIDDLE_LAYERS, l_middle_size=L_MIDDLE_SIZE):
    return (
            [random_weights(L0_SIZE, l_middle_size)] +
            [random_weights(l_middle_size, l_middle_size) for _ in range(n_middle_layers)] +
            [random_weights(l_middle_size, L_END_SIZE)]
    )


NEURON_TO_ACTION = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}


def nn_action_selection_for_input(nn: np.ndarray, input_activation: np.ndarray):
    activation = input_activation
    for layer in nn:
        activation = np.tanh(np.matmul(activation, layer))
    maximal_activation_neuron = int(np.argmax(activation))
    return NEURON_TO_ACTION[maximal_activation_neuron]


def get_catie_param(catie_agent):
    trend = catie_agent.trend
    p_explore = catie_agent.get_p_explore()
    last_choice = catie_agent.get_last_choice()
    biased_ca, anti_biased_ca = catie_agent.get_contingent_average()
    return np.array([trend, p_explore, last_choice, biased_ca, anti_biased_ca])


def get_input_activation(trial_number=0, allocated_reward_target=0,
                         allocated_reward_anti_target=0, catie_agent=CatieAgent(NUMBER_OF_TRIALS)):
    intercept = [1]
    a = np.concatenate((intercept, [trial_number, allocated_reward_target, allocated_reward_anti_target],
                        get_catie_param(catie_agent)))
    return a


def constrain_one_side(allocation, r_remain, trial_number):
    """
    Enforce exactly 25 rewards per 100 trials.
    """
    if r_remain == 0:
        return 0
    elif (NUMBER_OF_TRIALS - (trial_number + 1)) - r_remain == 0:  # + 1 because trials are 0..99
        return 1
    else:
        return allocation


def constrain_allocation(target_allocation, anti_targe_allocation, r_target_remain, r_anti_target_remain, trial_number):
    return (constrain_one_side(target_allocation, r_target_remain, trial_number),
            constrain_one_side(anti_targe_allocation, r_anti_target_remain, trial_number))


def network_score_single_run(nn):
    r_target_remain, r_anti_target_remain = 25, 25
    catie_agent = CatieAgent(NUMBER_OF_TRIALS)
    for trial_number in range(NUMBER_OF_TRIALS):
        # Allocate
        input_activation = get_input_activation(trial_number, r_target_remain, r_anti_target_remain, catie_agent)
        target_allocation, anti_targe_allocation = nn_action_selection_for_input(nn, input_activation)
        target_allocation, anti_targe_allocation = constrain_allocation(target_allocation, anti_targe_allocation,
                                                                        r_target_remain, r_anti_target_remain,
                                                                        trial_number)
        r_target_remain -= target_allocation
        r_anti_target_remain -= anti_targe_allocation

        # Apply choice
        choice = catie_agent.choose()
        outcome = target_allocation, anti_targe_allocation
        catie_agent.receive_outcome(choice, outcome)
    return catie_agent.previous_choices


def network_multiple_runs(nn, n_runs=10):
    biases = []
    for _ in range(n_runs):
        choices = network_score_single_run(nn)
        biases.append(choices.sum())
    return biases, np.mean(biases)


MUTATION_BASE_RATE = 0.005  # Arbitrary, greater numbers induce greater mutation for each generation


def mutate(layer, noise_magnitude=MUTATION_BASE_RATE):
    """
    Return the layer with added noise~U(-0.05, 0.05)
    """
    noise = (np.random.rand(*np.shape(layer)) - 0.5) * noise_magnitude
    return layer + noise


def crossover_and_mutate(nn1, nn2, progress, mutate_rate=MUTATION_BASE_RATE):
    """
    Return the outcome of crossing (choosing weights randomly from each nn) and
    mutating the outcome (just inserting noise).
    """
    output_nn = []
    for (layer_i_1, layer_i_2) in zip(nn1, nn2):
        # Crossover
        indices_from_n1 = np.random.randint(low=0, high=2, size=np.shape(layer_i_1))
        crossover_layer_i = np.where(indices_from_n1, layer_i_1, layer_i_2)
        # Mutate
        mutation_magnitude = mutate_rate * (1 - progress)
        crossover_mutated_output_layer_i = mutate(crossover_layer_i, mutation_magnitude)
        output_nn.append(crossover_mutated_output_layer_i)
    return output_nn


def get_next_generation(nn1, nn2, generation_size, progress, mutate_rate):
    """
    Next generation is made of:
      1. One new random set of weights
      2. The previous 2 best sets of weights (nn1 and nn2)
      3. gneeration_size-3 (currently 10-3=7) sets of weights that are the random
        combination of the two previously best networks (nn1 and nn2) on which a
        slight random modulation ("mutation") is applied.

      progress is a variable between 0 and 1 representing what proportion of the
      optimization has been completed (0 indicates just started, 1 means ended)
    """
    return [init_nn(), nn1, nn2] + [crossover_and_mutate(nn1, nn2, progress, mutate_rate) for _ in
                                    range(generation_size - 3)]


# Where should the data be saved
NETWORK_NAME = '6 layers network different mutation rate'
OUTPUT_PATH = "Data/" + NETWORK_NAME.replace(" ", "_") + ".pickle"


def save_network(best_nn, iteration, repetitions, generation_means):
    what_to_save = [best_nn, iteration, repetitions, generation_means]
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(what_to_save, f)


def load_network():
    pickle_file = open(OUTPUT_PATH, 'rb')
    (best_nn, iteration, repetitions, generation_means) = pickle.load(pickle_file)
    return best_nn, iteration, generation_means


TEST_REPETITION = 1000
DEBUG = False
if DEBUG:
    MIN_OPTIMIZATION_PER_NN = 2
    MAX_OPTIMIZATION_PER_NN = 5
    GENERATION_SIZE = 4
    NUM_OPTIMIZATION_TRIALS = 4
else:
    MIN_OPTIMIZATION_PER_NN = 100
    MAX_OPTIMIZATION_PER_NN = 1000
    GENERATION_SIZE = 10
    NUM_OPTIMIZATION_TRIALS = 1000

OPTIMIZATION_PER_REPETITION = np.linspace(MIN_OPTIMIZATION_PER_NN, MAX_OPTIMIZATION_PER_NN, num=NUM_OPTIMIZATION_TRIALS)


def get_two_best_networks(networks_scores):
    return heapq.nlargest(2, range(len(networks_scores)), key=lambda x: networks_scores[x][1])


def run_optimization(mutate_rate, generation, iteration_start, generation_mean_bias=None):
    t = tqdm.trange(iteration_start, NUM_OPTIMIZATION_TRIALS, desc="Iteration")
    best_nn = None

    if generation_mean_bias is None:
        generation_mean_bias = np.array([np.arange(0, NUM_OPTIMIZATION_TRIALS), np.zeros(NUM_OPTIMIZATION_TRIALS)])

    for i in t:
        optimization_trials = int(OPTIMIZATION_PER_REPETITION[i])
        # List of network [bias,mean_bias] for each network in generation
        networks_scores = [network_multiple_runs(nn, optimization_trials) for nn in generation]
        # Use heap to get two most fit networks indexes
        best_two_nn_indices = get_two_best_networks(networks_scores)
        # Best network in current generation
        best_nn = generation[best_two_nn_indices[0]]
        bias_distribution_nn_mean = networks_scores[best_two_nn_indices[0]][1]
        # Used for tracking progress while training
        t.set_description(f"best_nn mean = {bias_distribution_nn_mean} Optimization trials = {optimization_trials}",
                          refresh=True)
        generation_mean_bias[1][i] = bias_distribution_nn_mean
        save_network(best_nn, i, optimization_trials, generation_mean_bias)
        generation = get_next_generation(nn1=generation[best_two_nn_indices[0]],
                                         nn2=generation[best_two_nn_indices[1]],
                                         generation_size=GENERATION_SIZE,
                                         progress=i / NUM_OPTIMIZATION_TRIALS, mutate_rate=mutate_rate)
    return best_nn, generation_mean_bias


def continue_optimization(mutate_rate):
    best_nn, iteration, generation_mean_bias = load_network()
    generation = [best_nn] + [init_nn() for _ in range(GENERATION_SIZE - 1)]
    return run_optimization(mutate_rate, generation, iteration)


def run_optimization_from_scratch(mutate_rate):
    generation = [init_nn() for _ in range(GENERATION_SIZE)]
    return run_optimization(mutate_rate, generation, 0)


def plot_distribution(dist, color, dist_type):
    plt.hist(dist, alpha=0.5, density=True, bins=20, facecolor=color)
    plt.axvline(statistics.mean(dist), linestyle='dashed', linewidth=1, color=color)
    dist_mean = statistics.mean(dist)
    print(dist_type, ':', dist_mean)
    return dist, dist_mean


def get_static_catie_opt():
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
    return sequence_catie_score(optimized, 1000)


def main():
    fig, ax = plt.subplots()
    color_sequence = ['#ff7f0e']
    mutation_rates = [0.005]
    for i, mutate_rate in enumerate(mutation_rates):
        # best_nn, generation_mean_bias = run_optimization_from_scratch(mutate_rate)
        best_nn, generation_mean_bias = continue_optimization(mutate_rate)
        bias_distribution_best_nn, mean_bias_best_nn = network_multiple_runs(best_nn, TEST_REPETITION)
        ax.plot(generation_mean_bias[0], generation_mean_bias[1], color=color_sequence[i])
        mutation_rates[i] = (mean_bias_best_nn,
                             f'error: +/-{statistics.pstdev(bias_distribution_best_nn) / TEST_REPETITION:.3f}%')
    ax.legend(mutation_rates)
    ax.set(xlabel='Generation', ylabel='Mean Generation Bias (%)')
    ax.grid()
    fig.show()
    plt.figure()
    bias_distribution_naive_optimal_static = plot_schedules(bias_distribution_best_nn)
    plt.show()


def plot_schedules(bias_distribution_best_nn):
    # Plot Dynamic schedule histogram
    dist, dist_mean = plot_distribution(bias_distribution_best_nn, [x / 255 for x in (102, 196, 197)], 'Dynamic')
    plt.text(statistics.mean(dist) * 1.1, plt.ylim()[1] * 0.9, 'mean: {:.2f}'.format(dist_mean))
    # Plot static schedule histogram
    bias_distribution_naive_optimal_static, bias_distribution_naive_optimal_static_mean = get_static_catie_opt()
    dist, dist_mean = plot_distribution(bias_distribution_naive_optimal_static, [x / 255 for x in (238, 50, 51)],
                                        'Static')
    plt.text(statistics.mean(dist) * 1.1, plt.ylim()[1] * 0.8, 'mean: {:.2f}'.format(dist_mean))
    plt.ylabel('Probability')
    plt.xlabel('Bias')
    plt.legend(('Dynamic', 'Static', 'Dynamic', 'Static'), loc='upper left')
    print(scipy.stats.ttest_ind(bias_distribution_best_nn, bias_distribution_naive_optimal_static))
    plt.show()
    return bias_distribution_naive_optimal_static


if __name__ == '__main__':
    main()
