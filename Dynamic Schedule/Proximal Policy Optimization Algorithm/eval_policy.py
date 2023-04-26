"""
This file is used only to evaluate our trained policy/actor after
training in main.py with ppo.py. I wrote this file to demonstrate
that our trained policy exists independently of our learning algorithm,
which resides in ppo.py. Thus, we can test our trained policy without
relying on ppo.py.
"""
import torch
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
    add_mask = torch.FloatTensor([1e-9, 0.0, 0.0, 0.0])

    # Create masks for valid actions
    mask_0_condition = (trial_number > N_TRIALS - N_REWARDS_PER_ALT) & ((assignments[0] == N_TRIALS - trial_number) |
                                                                        (assignments[1] == N_TRIALS - trial_number))
    mask_1_condition = assignments[0] < N_REWARDS_PER_ALT
    mask_2_condition = assignments[1] < N_REWARDS_PER_ALT

    mask[0] *= (~mask_0_condition)
    mask[1] *= mask_1_condition
    mask[2] *= mask_2_condition
    mask[3] *= (mask_1_condition & mask_2_condition)

    # Apply the mask to the action probabilities
    constrained_probs = (action_probs + add_mask) * mask
    constrained_probs /= constrained_probs.sum()

    # Sample an action from the distribution
    action_idx = torch.multinomial(constrained_probs, num_samples=1).item()
    action = INDEX_TO_ACTION[action_idx]

    # Return the sampled action and the log probability of that action in our distribution
    return action


def rollout(policy, env):
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
    # Rollout until user kills process
    while True:
        # Reset the environment. sNote that obs is short for observation.
        observation, _ = env.reset()
        for _ in range(N_TRIALS):
            # Calculate action and make a step in the env.
            action = select_action(observation, policy)
            observation, reward, done, _ = env.step(action)
            # If the environment tells us the episode is terminated, break
            if done:
                break
        # returns episodic length and return in this iteration
        yield env.compute_reward()


def eval_policy(policy, env, num_iterations=10000):
    """
        The main function to evaluate our policy with. It will iterate a generator object
        "rollout", which will simulate each episode and return the most recent episode's
        length and return. We can then log it right after. And yes, eval_policy will run
        forever until you kill the process.

        Parameters:
            policy - The trained policy to test, basically another name for our actor model
            env - The environment to test the policy on
            render - Whether we should render our episodes. False by default.

        Return:
            None

        NOTE: To learn more about generators, look at rollout's function description
    """
    # Rollout with the policy and environment, and log each episode's data
    ep_returns = []
    for ep_num, ep_ret in tqdm(enumerate(rollout(policy, env))):
        ep_returns.append(ep_ret)
        if ep_num >= num_iterations - 1:
            break
    return ep_returns
