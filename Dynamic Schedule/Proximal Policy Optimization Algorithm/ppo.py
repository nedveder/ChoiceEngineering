import os
import random
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import Adam
import gymnasium as gym
from network import ForwardNet

INDEX_TO_ACTION = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}
ACTION_TO_INDEX = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}


class PPO:
    """
    This is the PPO class we will use as our model in main.py
    """
    # Used for plotting network training over time.
    network_rewards_each_epoch = []
    network_error_each_epoch = []

    def __init__(self, env, n_episodes, n_trials, n_repetitions, hidden_size,
                 **hyperparameters):
        """
            Initializes the PPO model, including hyperparameters.
            Parameters:
                env : the environment to train on.
                n_episodes : number of iterations to preform training loop.
                n_trials : number of trials in each episode for experiment.
                n_repetitions : number of repetitions to preform test for network,used for benchmarking
                hidden_size : size of each hidden layer of actor and critic networks
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

            Returns:
                None
        """
        # Make sure the environment is compatible with our code
        assert (type(env.observation_space) == gym.spaces.Box)
        assert (type(env.action_space) == gym.spaces.Box)

        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.n_episodes = n_episodes
        self.n_trials = n_trials
        self.n_repetitions = n_repetitions
        self.n_updates_per_batch = 10
        self.hidden_size = hidden_size
        self.max_bias_achieved = 0

        # Initialize actor and critic networks
        self.actor = ForwardNet(self.obs_dim, self.hidden_size, self.act_dim)
        self.critic = ForwardNet(self.obs_dim, self.hidden_size, 1)

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # This logger will help us with printing out summaries of each batch
        self.logger = {
            'delta_t': time.time_ns(),
            'cur_batch': 0,
            'n_batches': 0,
            'batch_rewards': [],  # episodic returns in batch
            'actor_losses': [],  # losses of actor network in current iteration
            'start_training': time.time_ns()
        }

    def learn(self, n_batches):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.

            Parameters:
                n_batches - the total number of batches to train for, each batch consists of n_episodes and each episode
                    consists of n_trials.

            Return:
                None
        """
        print(f"Learning... Running {self.n_trials} trials per episode, ", end='')
        print(f"{self.n_episodes} episodes per batch for a total of {n_batches} batches")
        self.logger['n_batches'] = n_batches
        for batch_num in range(n_batches):

            self.logger['cur_batch'] = batch_num
            # Get data from environment for batch training
            batch_obs, batch_acts, batch_log_probs, batch_rtgs = self.rollout()

            # Calculate advantage at k-th iteration
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_t = batch_rtgs - V.detach()

            # Normalizing advantages isn't theoretically necessary, but in practice it decreases the variance of
            # our advantages and makes convergence much more stable and faster.
            A_t = (A_t - A_t.mean()) / (A_t.std() + 1e-10)

            # This is the loop where we update our network for n_updates_per_iteration
            for _ in range(self.n_updates_per_batch):
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                # NOTE: we just subtract the logs, which is the same as
                # dividing the values and then canceling the log with e^log.
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_t
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_t

                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                # Will not delete intermediary results, so we can preform backward pass once again for critic
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach())

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if batch_num % self.save_freq == 0:
                self.plot_training_progress()
                # Make sure to only save new network if it's an improvement over previous.
                if self.network_rewards_each_epoch[-1] > self.max_bias_achieved:
                    self.max_bias_achieved = self.network_rewards_each_epoch[-1]
                    torch.save(self.actor.state_dict(), './ppo_actor.pth')
                    torch.save(self.critic.state_dict(), './ppo_critic.pth')

    def rollout(self):
        """

        """
        # Batch data.
        batch_obs = np.zeros((self.n_episodes * self.n_trials, self.obs_dim))
        batch_acts = np.zeros((self.n_episodes * self.n_trials, 2))
        batch_log_probs = np.zeros(self.n_episodes * self.n_trials)
        batch_rewards = np.zeros((self.n_episodes, self.n_trials))
        ep_rewards = torch.zeros(self.n_trials)

        for episode in range(self.n_episodes):
            ep_rewards = ep_rewards.detach() * 0
            # Reset the environment. sNote that obs is short for observation.
            observation, _ = self.env.reset()

            for trial in range(self.n_trials):

                # Track observations in this batch
                batch_obs[episode * self.n_trials + trial] = observation

                # Calculate action and make a step in the env.
                action, log_prob = self.select_action(observation)
                observation, reward, done, _ = self.env.step(action)

                # Track recent reward, action, and action log probability
                ep_rewards[trial] = reward
                batch_acts[episode * self.n_trials + trial] = action
                batch_log_probs[episode * self.n_trials + trial] = log_prob

                # If the environment tells us the episode is terminated, break
                if done:
                    break

            # Track episodic rewards
            batch_rewards[episode] = ep_rewards

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rewards)  # ALG STEP 4

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rewards'] = list(batch_rewards)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs

    def compute_rtgs(self, batch_rewards):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.
            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)
            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # Convert batch_rewards to a PyTorch tensor
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float)

        # Initialize an empty tensor to store the rewards-to-go
        batch_rtgs = torch.empty_like(batch_rewards)

        # Iterate through each episode
        for i in range(batch_rewards.shape[0]):
            ep_rewards = batch_rewards[i, :]

            # Compute the rewards-to-go for the current episode
            rtg = torch.empty_like(ep_rewards)
            discounted_reward = 0
            for j in reversed(range(ep_rewards.shape[0])):
                discounted_reward = ep_rewards[j] + discounted_reward * self.gamma
                rtg[j] = discounted_reward

            # Store the computed rewards-to-go in the batch_rtgs tensor
            batch_rtgs[i, :] = rtg

        # Flatten the batch_rtgs tensor
        batch_rtgs = batch_rtgs.flatten()

        return batch_rtgs

    def select_action(self, state):
        action_probs = self.actor(state)

        assignments, trial_number = (state[9], state[10]), state[11]  # State indices for assignments
        # Create a mask for valid actions - CONSTRAINTS
        mask = torch.FloatTensor([1.0, 1.0, 1.0, 1.0])
        add_mask = torch.FloatTensor([1e-9, 0.0, 0.0, 0.0])

        # Apply constraints
        mask[0] = 0 if trial_number > 75 and (
                assignments[0] == 100 - trial_number or assignments[1] == 100 - trial_number) else 1.0
        mask[1] = 0 if assignments[0] >= 25 else 1.0
        mask[2] = 0 if assignments[1] >= 25 else 1.0
        mask[3] = 0 if assignments[0] >= 25 or assignments[1] >= 25 else 1.0

        # Apply the mask to the action probabilities
        constrained_probs = ((action_probs + add_mask) * mask) / (
            ((action_probs + add_mask) * mask).sum())
        # Create a distribution with the mean action and std from the covariance matrix above.
        # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # https://www.youtube.com/watch?v=JjB58InuTqM

        # Sample an action from the distribution
        action_idx = torch.multinomial(constrained_probs, num_samples=1).item()
        action = INDEX_TO_ACTION[action_idx]

        # Calculate the log probability for that action
        log_prob = torch.log(constrained_probs[ACTION_TO_INDEX[action]])

        # Return the sampled action and the log probability of that action in our distribution
        return action, log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        """
        Estimate the values of each observation, and the log probs of
        each action in the most recent batch with the most recent
        iteration of the actor network. Should be called from learn.

        Parameters:
            batch_obs : the observations from the most recently collected batch as a tensor.
                        Shape: (number of trials in a batch, dimension of observation)
            batch_acts : the actions from the most recently collected batch as a tensor.
                        Shape: (number of trials in a batch, dimension of action)

        Return:
            V : the predicted values of batch_obs
            log_probs : the log probabilities of the actions taken in batch_acts given batch_obs
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in select_action()
        gaussian_action_mean = self.actor(batch_obs)

        batch_size = batch_obs.shape[0]

        assignments, trial_numbers = batch_obs[:, 9:11], batch_obs[:, 11]

        # Create masks for valid actions - CONSTRAINTS
        mask = torch.FloatTensor([1.0, 1.0, 1.0, 1.0]).repeat(batch_size, 1)
        add_mask = torch.FloatTensor([1e-9, 0.0, 0.0, 0.0]).repeat(batch_size, 1)

        # Apply constraints
        mask[(trial_numbers > 75) & (
                (assignments[:, 0] == (100 - trial_numbers)) | (assignments[:, 1] == (100 - trial_numbers))), 0] = 0
        mask[assignments[:, 0] >= 25, 1] = 0
        mask[assignments[:, 0] >= 25, 3] = 0
        mask[assignments[:, 1] >= 25, 2] = 0
        mask[assignments[:, 1] >= 25, 3] = 0

        # Compute constrained_probs for the entire batch
        constrained_probs = ((gaussian_action_mean + add_mask) * mask) / (
            ((gaussian_action_mean + add_mask) * mask).sum(dim=1, keepdim=True))

        # Convert batch_acts to integer indices
        batch_acts_indices = torch.tensor(
            [ACTION_TO_INDEX[tuple(int(x) for x in act.tolist())] for act in batch_acts], dtype=torch.long)

        # Compute log probabilities for the entire batch
        log_probs = torch.log(constrained_probs[torch.arange(batch_size), batch_acts_indices])

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs

    def plot_training_progress(self) -> None:
        """
        Plot the training progress of the agent, including episode rewards.
        test_network is run using only policy network and
        This function generates a live plot of episode rewards as the training progresses.

        Args:

        Returns:
            None
        """
        episode_rewards = self._test_network()
        self.network_rewards_each_epoch.append(episode_rewards.mean())
        self.network_error_each_epoch.append(np.std(episode_rewards) / np.sqrt(self.n_repetitions))
        epoch = len(self.network_rewards_each_epoch)
        # Update the live plot
        x = list(range(epoch))

        # Perform linear regression to find the best-fit line
        if epoch > 1:
            fit_coeffs = np.polyfit(x, self.network_rewards_each_epoch, 1)
            fit_poly = np.poly1d(fit_coeffs)
            y_fit = fit_poly(x)

        # Plot the data with the best-fit line
        fig, ax = plt.subplots()
        ax.plot(x, self.network_rewards_each_epoch, 'royalblue', label='Epoch Rewards')
        if epoch > 1:
            ax.plot(x, y_fit, 'r', linestyle='--', label='Best-fit Line')
        ax.set_ylabel('Epoch Rewards')
        ax.set_xlabel(f'Epoch(Batches % {self.save_freq})')

        # Plot the confidence interval with light blue color
        lower_bounds = np.array(self.network_rewards_each_epoch) - np.array(self.network_error_each_epoch)
        upper_bounds = np.array(self.network_rewards_each_epoch) + np.array(self.network_error_each_epoch)
        ax.fill_between(x, lower_bounds, upper_bounds, color='lightblue', alpha=0.5, label='Confidence Interval')

        # Add line for the highest reward mean yet
        max_reward = max(self.network_rewards_each_epoch)
        ax.hlines(max_reward, 0, epoch - 1, colors='g', linestyle='dotted', label='Highest Reward Mean')

        # Add annotations
        ax.set_title("Policy Network Training for Catie Agent")
        if epoch > 1:
            ax.annotate(f"Training Slope: {fit_coeffs[0]:.2f}", xy=(0.5, 0.45), xycoords='axes fraction')
        ax.annotate(f"N = {self.n_repetitions}", xy=(0.5, 0.4), xycoords='axes fraction')
        ax.annotate(f"Highest Reward Mean = {max_reward:.2f}", xy=(0.5, 0.35), xycoords='axes fraction')
        ax.legend()
        plt.savefig("network_fitness.png")

        # Append data to file
        plot_data = {
            'cur_epoch': [epoch],
            'network_rewards_each_epoch': [self.network_rewards_each_epoch[-1]],
            'network_error_each_epoch': [self.network_error_each_epoch[-1]],
        }
        batch_df = pd.DataFrame(plot_data)
        # Save the batch data to a CSV file
        csv_file = 'plot_data.csv'
        batch_df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

    def _test_network(self):
        rep_rewards = np.zeros(self.n_repetitions)
        # REPETITIONS FOR INCREASED ACCURACY IN RESULTS
        for repetition in range(self.n_repetitions):
            state, _ = self.env.reset()
            for _ in range(self.n_trials):
                action, log_action_prob = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                if done:
                    break
                else:
                    state = next_state
            rep_rewards[repetition] = self.env.compute_reward()
        return rep_rewards

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters

            Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                    hyperparameters defined below with custom values.

            Return:
                None
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.lr = 0.005  # Learning rate of actor optimizer
        self.gamma = 0.95  # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2  # Recommended 0.2, helps define the threshold to clip the ratio during SGA

        # Miscellaneous parameters
        self.save_freq = 10  # How often we save in number of iterations
        self.seed = 1  # Sets the seed of our program, used for reproducibility of results

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        # Sets the seed if specified
        if self.seed is not None:
            # Check if our seed is valid first
            assert (type(self.seed) == int)

            # Set the seed
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch.

            Parameters:

            Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        total_time = (time.time_ns() - self.logger['start_training']) / 1e9

        cur_batch = self.logger['cur_batch']
        avg_ep_rewards = np.mean([np.sum(ep_rewards) for ep_rewards in self.logger['batch_rewards']])
        avg_ep_error = np.mean(
            [np.std(ep_rewards) / np.sqrt(len(ep_rewards)) for ep_rewards in self.logger['batch_rewards']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        # Print logging statements
        print(flush=True)
        print(f"---------------- Batches so far {cur_batch}/{self.logger['n_batches']} ------------------", flush=True)
        print(f"Average Episodic Return: {avg_ep_rewards:.2f}", flush=True)
        print(f"Average Episodic Return Standard Error: {avg_ep_error:.2f}", flush=True)
        print(f"Average Loss: {avg_actor_loss:.5f}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"Total time: {total_time:.2f} secs", flush=True)
        print(flush=True)

        # Append data to file
        epoch_data = {
            'cur_batch': [cur_batch],
            'avg_ep_rewards': [avg_ep_rewards],
            'avg_ep_error': [avg_ep_error],
            'avg_actor_loss': [avg_actor_loss],
        }
        batch_df = pd.DataFrame(epoch_data)
        # Save the batch data to a CSV file
        csv_file = 'epoch_data.csv'
        batch_df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

        # Reset batch-specific logging data
        self.logger['cur_batch'] = 0
        self.logger['batch_rewards'] = []
        self.logger['actor_losses'] = []