"""
This file is used only to evaluate our trained policy/actor after
training in main.py with ppo.py. I wrote this file to demonstrate
that our trained policy exists independently of our learning algorithm,
which resides in ppo.py. Thus, we can test our trained policy without
relying on ppo.py.
"""
import matplotlib.font_manager
import numpy as np
import torch
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from scipy.stats import norm

from tqdm import tqdm

N_REWARDS_PER_ALT = 25
N_TRIALS = 100

INDEX_TO_ACTION = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}
ACTION_TO_INDEX = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}


def _log_summary(ep_ret, ep_num):
    """
        Print to stdout what we've logged so far in the most recent episode.

        Parameters:

        Return:
            None
    """
    # Print logging statements
    print(flush=True)
    print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
    print(f"Episodic Return: {ep_ret:.2f}", flush=True)
    print(flush=True)


def select_action(state, policy):
    """
    Select an action from the given policy for a given state while considering constraints.

    This function samples an action from the action probabilities produced by the policy for the current state.
    It also applies constraints on the actions based on the specific problem domain. It returns the sampled action
    and the log probability of that action.

    Parameters:
    -----------
    state : torch.tensor
        A tensor representing the current state of the environment.
    policy : Policy class instance
        An instance of the policy class, which provides the action probabilities for the given state.

    Returns:
    --------
    action : int
        The sampled action to be taken in the current state, considering the problem-specific constraints.
    """
    # Get action probabilities from network
    action_probs = policy(state, is_test=True)

    # Return the sampled action
    return INDEX_TO_ACTION[torch.multinomial(action_probs, num_samples=1).item()]


def rollout(policy, env, n_iterations):
    episode_biases = np.zeros(n_iterations)
    episode_choices = np.zeros((n_iterations, 100))
    episode_actions = np.zeros((n_iterations, N_TRIALS, 2))
    choices = np.zeros(N_TRIALS)
    actions = np.zeros((N_TRIALS, 2))

    for index in tqdm(range(n_iterations)):
        observation, _ = env.reset()
        choices *= 0
        actions *= 0
        for i in range(N_TRIALS):
            action_probs = policy(observation, is_test=True)
            action = INDEX_TO_ACTION[torch.multinomial(action_probs, num_samples=1).item()]

            observation, reward, done, choice = env.step(action)
            choices[i] = choice
            actions[i] = action

        episode_biases[index] = env.compute_reward()
        episode_choices[index] = choices
        episode_actions[index] = actions

    return episode_biases, episode_choices, episode_actions


def eval_policy(policy, env, n_iterations, name):
    """
        The main function to evaluate our policy with. It will plot different information on the choices made and biases
        received.

        Parameters:
            policy - The trained policy to test, basically another name for our actor model
            env - The environment to test the policy on
            n_iterations -Number of iterations to rollout data for.

        Return:
            None
    """
    # Rollout with the policy and environment, and log each episode's data
    ep_bias, ep_agent_choices, ep_actions = rollout(policy, env, n_iterations)
    plot_data(ep_bias, ep_agent_choices, ep_actions, str(name))


def plot_data(ep_bias, ep_choices, ep_actions, name):
    """
    Plots the Bias distribution, Average reward assignment, and Per trial average choice probability.

    Parameters:
        ep_bias : A list of the total bias for each experiment.
        ep_choices : A list of lists, where every sublist describes the choices for each trial in the episode.
        ep_actions : A list of lists, where every sublist describes the actions(reward allocation) meaning a list
            of tuples for each trial in the episode.
        name : Name of the current network

    Return:
        None
    """
    # 1. Bias distribution
    plt.figure()
    mean_ep_returns = np.mean(ep_bias)
    std_error = np.std(ep_bias) / np.sqrt(len(ep_bias))
    x_values = np.arange(0, 100)

    # Create a histogram using the first y-axis
    ax1 = plt.gca()
    ax1.hist(ep_bias, bins=30, range=(0, 100), density=True, color='blue', alpha=0.5)
    ax1.set_ylabel('Frequency', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a second y-axis
    ax2 = ax1.twinx()

    pdf_values = norm.pdf(x_values, loc=mean_ep_returns, scale=np.std(ep_bias))
    # Plot the probability distribution using the second y-axis
    ax2.plot(x_values, pdf_values, label='Probability Distribution', color='red')
    ax2.set_ylabel('Probability', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.xlabel('Bias')
    plt.title('Bias Distribution')

    # Add an annotation for the mean and standard error
    annotation_text = f"Mean: {mean_ep_returns:.2f}(+-{std_error:.2f})"
    plt.annotate(annotation_text, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=12,
                 bbox=dict(facecolor='white'))

    # Save the figure
    plt.savefig(f'Plots/{name}/bias_distribution.png')

    # 2. Average reward assignment
    plt.figure(figsize=(16, 3.8))
    reward_probabilities = np.mean(ep_actions, axis=0)
    reward_probabilities_error = np.std(ep_actions, axis=0) / np.sqrt(len(ep_actions))

    for trial, (prob_1, prob_0) in enumerate(reward_probabilities):
        plt.scatter([trial], [0.48], c=[prob_0], cmap='Oranges', vmin=0, vmax=1, s=140, edgecolors='gray' \
            if prob_0 == 0 else 'black')
        plt.scatter([trial], [0.52], c=[prob_1], cmap='Oranges', vmin=0, vmax=1, s=140, edgecolors='gray' \
            if prob_1 == 0 else 'black')

    plt.yticks([0.48, 0.52], ['Alternative 2', 'Alternative 1'],
               fontproperties=matplotlib.font_manager.FontProperties(size=14))
    plt.ylim(0.45, 0.55)  # Adjust the y-axis limits
    plt.xlabel('Trial')
    plt.title('Reward Probability per Alternative per Trial')
    plt.colorbar(label='P(Reward)')

    plt.savefig(f'Plots/{name}/average_reward_assignment.png')

    # 2.1
    plt.figure()
    # Creating the x-axis values for the sub-arrays
    x1 = np.arange(0, 100)
    x2 = np.arange(0, 100)

    # Plotting the sub-arrays
    plt.errorbar(x1, reward_probabilities[:, 0], yerr=reward_probabilities_error[:, 0], label="Alternative A",
                 linestyle='-', marker='o', ecolor='b', color='royalblue', alpha=0.5)
    plt.errorbar(x2, reward_probabilities[:, 1], yerr=reward_probabilities_error[:, 1], label="Alternative B",
                 linestyle='-', marker='o', ecolor='r', color='maroon', alpha=0.1)

    # Customizing the plot
    plt.xlabel("Trial")
    plt.ylabel("Probability Density")
    plt.annotate(f"Total rewards given:{np.sum(reward_probabilities[:, 0],axis=0)},"
                 f"{np.sum(reward_probabilities[:, 1],axis=0)}", xy=(0.1, 1.08), xycoords='axes fraction')

    plt.title("Probability for Reward Assignment")
    plt.legend()

    plt.savefig(f'Plots/{name}/average_reward_assignment_pdf.png')

    # 3. Per trial average choice probability
    plt.figure()
    n_iterations = len(ep_choices)
    choice_probabilities = [sum(trial) / n_iterations for trial in zip(*ep_choices)]
    plt.plot(choice_probabilities)
    plt.xlabel('Trial')
    plt.ylabel('Average Choice Probability')
    plt.title('Per Trial Average Choice Probability')
    plt.savefig(f'Plots/{name}/per_trial_average_choice_probability.png')

    plt.close('all')
