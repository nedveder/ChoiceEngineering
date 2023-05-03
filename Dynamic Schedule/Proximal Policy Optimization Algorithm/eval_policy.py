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
    action_probs = policy(state)

    # Sample an action from the distribution
    action_idx = torch.multinomial(action_probs, num_samples=1).item()
    action = INDEX_TO_ACTION[action_idx]

    # Return the sampled action and the log probability of that action in our distribution
    return action


def rollout_single_episode(policy, env):
    observation, _ = env.reset()
    rewards = []
    actions = []
    for _ in range(N_TRIALS):
        action = select_action(observation, policy)
        observation, reward, done, _ = env.step(action)
        rewards.append(reward)
        actions.append(action)
    episode_bias = env.compute_reward()
    return episode_bias, rewards, actions


def parallel_rollout(policy, env, n_iterations):
    episode_biases = np.zeros(n_iterations)
    episode_rewards = [None] * n_iterations
    episode_actions = [None] * n_iterations

    with ProcessPoolExecutor() as executor:
        future_results = [executor.submit(rollout_single_episode, policy, env) for _ in range(n_iterations)]

        for future in concurrent.futures.as_completed(future_results):
            index = future_results.index(future)
            episode_bias, rewards, actions = future.result()
            episode_biases[index] = episode_bias
            episode_rewards[index] = rewards
            episode_actions[index] = actions

    return episode_biases, episode_rewards, episode_actions


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
    ep_bias, ep_agent_choices, ep_actions = parallel_rollout(policy, env, n_iterations)
    plot_data(ep_bias, ep_agent_choices, ep_actions, str(name))


def plot_data(ep_bias, ep_choices, ep_actions, name):
    """
    Plots the Bias distribution, Average reward assignment, and Per trial average choice probability.

    Parameters:
        ep_bias : A list of the total bias for each experiment.
        ep_choices (list): A list of lists, where every sublist describes the choices for each trial in the episode.
        ep_actions (list): A list of lists, where every sublist describes the actions(reward allocation) meaning a list
            of tuples for each trial in the episode.
        name : Name of the current network

    Return:
        None
    """
    # 1. Bias distribution
    plt.figure()
    plt.hist(ep_bias, bins=30, density=True)
    plt.xlabel('Bias')
    plt.ylabel('Frequency')
    plt.title('Bias Distribution')

    mean_ep_returns = np.mean(ep_bias)
    std_error = np.std(ep_bias) / np.sqrt(len(ep_bias))
    # Add an annotation for the mean and standard error
    annotation_text = f"Mean: {mean_ep_returns:.2f}(+-{std_error:.2f})"
    plt.annotate(annotation_text, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=12,
                 bbox=dict(facecolor='white'))

    plt.savefig(f'Plots/{name}/bias_distribution_{name}.png')

    # 2. Average reward assignment
    plt.figure(figsize=(16, 3.8))
    reward_probabilities = np.mean(ep_actions, axis=0)

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

    plt.savefig(f'Plots/{name}/average_reward_assignment_{name}.png')

    # 3. Per trial average choice probability
    plt.figure()
    n_iterations = len(ep_choices)
    choice_probabilities = [sum(trial) / n_iterations for trial in zip(*ep_choices)]
    plt.plot(choice_probabilities)
    plt.xlabel('Trial')
    plt.ylabel('Average Choice Probability')
    plt.title('Per Trial Average Choice Probability')
    plt.savefig(f'Plots/{name}/per_trial_average_choice_probability_{name}.png')
