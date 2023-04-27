"""
This file is used only to evaluate our trained policy/actor after
training in main.py with ppo.py. I wrote this file to demonstrate
that our trained policy exists independently of our learning algorithm,
which resides in ppo.py. Thus, we can test our trained policy without
relying on ppo.py.
"""
import numpy as np
import torch
from matplotlib import pyplot as plt
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
    action_probs = policy(state)
    assignments, trial_number = (state[9], state[10]), state[11]  # State indices for assignments

    # Create a mask for valid actions - CONSTRAINTS
    mask = torch.ones(4)
    add_mask = torch.FloatTensor([0, 0.0, 0.0, 0.0])

    # Apply constraints
    mask[0] = 0 if trial_number > 75 and (
                assignments[0] >= 100 - trial_number or assignments[1] >= 100 - trial_number) else 1.0
    mask[1] = 0 if assignments[0] >= 25 else 1.0
    mask[2] = 0 if assignments[1] >= 25 else 1.0
    mask[3] = 0 if assignments[0] >= 25 or assignments[1] >= 25 else 1.0

    # Apply the mask to the action probabilities
    constrained_probs = (action_probs + add_mask) * mask
    constrained_probs /= constrained_probs.sum()

    # Sample an action from the distribution
    action_idx = torch.multinomial(constrained_probs, num_samples=1).item()
    action = INDEX_TO_ACTION[action_idx]

    # Return the sampled action and the log probability of that action in our distribution
    return action


def rollout(policy, env, n_iterations):
    """
        Returns a generator to roll out each episode given a trained policy and
        environment to test on.

        Parameters:
            policy - The trained policy to test
            env - The environment to evaluate the policy on
            render - Specifies whether to render or not

        Return:
            A generator object rollout, or iterable, which will return the latest
            episodic length and return on each iteration of the generator.

    """
    ep_bias = []
    ep_rewards = []
    ep_actions = []
    for _ in range(n_iterations):
        # Reset the environment. sNote that obs is short for observation.
        observation, _ = env.reset()
        # Because of the way the CATIE env is set up, the reward is the choice, Where 1 is the biased alternative, and 0
        # is the non-biased alternative.
        rewards = []
        actions = []
        for _ in range(N_TRIALS):
            # Calculate action and make a step in the env.
            action = select_action(observation, policy)
            observation, reward, done, _ = env.step(action)
            rewards.append(reward)
            actions.append(action)
        # returns episodic length and return in this iteration
        ep_bias.append(env.compute_reward())
        ep_rewards.append(rewards)
        ep_actions.append(actions)
    return ep_bias, ep_rewards, ep_actions


def eval_policy(policy, env, n_iterations):
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
    plot_data(ep_bias, ep_agent_choices, ep_actions)


def plot_data(ep_bias, ep_choices, ep_actions):
    """
    Plots the Bias distribution, Average reward assignment, and Per trial average choice probability.

    Parameters:
        ep_bias (list): A list of the total bias for each experiment.
        ep_choices (list): A list of lists, where every sublist describes the choices for each trial in the episode.
        ep_actions (list): A list of lists, where every sublist describes the actions(reward allocation) meaning a list
            of tuples for each trial in the episode.

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

    plt.savefig('bias_distribution.png')

    # 2. Average reward assignment
    plt.figure(figsize=(15, 2))
    reward_probabilities = np.mean(ep_actions, axis=0)

    for trial, (prob_1, prob_0) in enumerate(reward_probabilities):
        plt.scatter([trial] * 2, [0.4, 0.6], c=[prob_0, prob_1], cmap='coolwarm', vmin=0, vmax=1, s=50, edgecolors='black')

    plt.yticks([0.4, 0.6], ['Alternative 0', 'Alternative 1'])
    plt.ylim(0.3, 0.7)  # Adjust the y-axis limits
    plt.xlabel('Trial')
    plt.title('Reward Probability per Alternative per Trial')
    plt.colorbar(label='P(Reward)')
    plt.savefig('average_reward_assignment.png')

    # 3. Per trial average choice probability
    plt.figure()
    n_iterations = len(ep_choices)
    choice_probabilities = [sum(trial) / n_iterations for trial in zip(*ep_choices)]
    plt.plot(choice_probabilities)
    plt.xlabel('Trial')
    plt.ylabel('Average Choice Probability')
    plt.title('Per Trial Average Choice Probability')
    plt.savefig('per_trial_average_choice_probability.png')
