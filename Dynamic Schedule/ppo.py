import concurrent
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
from network import ForwardNet, DEVICE
from concurrent.futures import ProcessPoolExecutor

INDEX_TO_ACTION = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}
ACTION_TO_INDEX = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}


class PPO:
    """
    This is the PPO class we will use as our model in main.py
    """
    # Used for plotting network training over time.
    network_rewards_each_epoch = []
    network_error_each_epoch = []

    def __init__(self, env, n_episodes, n_trials, n_repetitions, hidden_size, **hyperparameters):
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
        self.name = hyperparameters['name']
        self.hidden_layers = 0
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
        self.actor = ForwardNet(self.obs_dim, self.hidden_layers, self.hidden_size, self.act_dim)
        self.critic = ForwardNet(self.obs_dim, self.hidden_layers, self.hidden_size, 1, critic=True)

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
            batch_obs, batch_acts, batch_log_probs, batch_rtgs = self.parallel_rollout()

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
                self.max_bias_achieved = self.network_rewards_each_epoch[-1]
                torch.save(self.actor.state_dict(), f'./{self.name}/ppo_actor_{self.name}.pth')
                torch.save(self.critic.state_dict(), f'./{self.name}/ppo_critic_{self.name}.pth')

    @staticmethod
    def rollout_single_episode(select_action, env, n_trials, obs_dim):
        ep_obs = np.zeros((n_trials, obs_dim))
        ep_acts = np.zeros((n_trials, 2))
        ep_log_probs = np.zeros(n_trials)
        ep_rewards = np.zeros(n_trials)

        observation, _ = env.reset()

        for trial in range(n_trials):
            ep_obs[trial] = observation
            action, log_prob = select_action(observation)
            observation, reward, done, _ = env.step(action)

            ep_rewards[trial] = reward
            ep_acts[trial] = action
            ep_log_probs[trial] = log_prob

            if done:
                break

        return ep_obs, ep_acts, ep_log_probs, ep_rewards

    def parallel_rollout(self):
        batch_size = self.n_episodes * self.n_trials
        batch_obs = np.zeros((batch_size, self.obs_dim))
        batch_acts = np.zeros((batch_size, 2))
        batch_log_probs = np.zeros(batch_size)
        batch_rewards = np.zeros((self.n_episodes, self.n_trials))

        with ProcessPoolExecutor() as executor:
            future_results = [
                executor.submit(self.rollout_single_episode, self.select_action, self.env, self.n_trials, self.obs_dim)
                for _ in range(self.n_episodes)]

            for future in concurrent.futures.as_completed(future_results):
                index = future_results.index(future)
                ep_obs, ep_acts, ep_log_probs, ep_rewards = future.result()
                episode_start = index * self.n_trials
                episode_end = (index + 1) * self.n_trials
                batch_obs[episode_start:episode_end] = ep_obs
                batch_acts[episode_start:episode_end] = ep_acts
                batch_log_probs[episode_start:episode_end] = ep_log_probs
                batch_rewards[index] = ep_rewards

        batch_obs = torch.tensor(batch_obs, dtype=torch.float, device=DEVICE)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float, device=DEVICE)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device=DEVICE)
        batch_rtgs = self.compute_rtgs(batch_rewards)

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

        action_idx = torch.multinomial(action_probs, num_samples=1).item()
        action = INDEX_TO_ACTION[action_idx]

        # Calculate the log probability for that action
        log_prob = torch.log(action_probs[ACTION_TO_INDEX[action]])
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
        action_probs = self.actor(batch_obs)
        batch_size = batch_obs.shape[0]

        # Convert batch_acts to integer indices
        batch_acts_indices = torch.tensor([ACTION_TO_INDEX[tuple[int, int](int(x) for x in act.tolist())]
                                           for act in batch_acts], dtype=torch.long)

        # Compute log probabilities for the entire batch
        log_probs = torch.log(action_probs[torch.arange(batch_size), batch_acts_indices])

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
        episode_rewards = self._parallel_test_network()
        self.network_rewards_each_epoch.append(episode_rewards.mean())
        self.network_error_each_epoch.append(np.std(episode_rewards) / np.sqrt(self.n_repetitions))
        epoch = len(self.network_rewards_each_epoch)
        # Update the live plot
        x = list(range(epoch))

        # Perform linear regression to find the best-fit line
        y_fit = []
        fit_coeffs = []
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
        ax.set_xlabel('Epoch')

        # Plot the confidence interval with light blue color
        lower_bounds = np.array(self.network_rewards_each_epoch) - np.array(self.network_error_each_epoch)
        upper_bounds = np.array(self.network_rewards_each_epoch) + np.array(self.network_error_each_epoch)
        ax.fill_between(x, lower_bounds, upper_bounds, color='lightblue', alpha=0.5, label='Confidence Interval')

        # Add line for the highest reward mean yet
        max_reward = max(self.network_rewards_each_epoch)
        ax.hlines(max_reward, 0, epoch - 1, colors='g', linestyle='dotted', label='Highest Reward Mean')

        # Add annotations
        ax.set_title("Policy Network Training for Catie Agent")
        xy_cord = 'axes fraction'
        if epoch > 1:
            ax.annotate(f"Training Slope: {fit_coeffs[0]:.2f}", xy=(0.5, 0.45), xycoords=xy_cord)
        ax.annotate(f"N = {self.n_repetitions}", xy=(0.5, 0.4), xycoords=xy_cord)
        ax.annotate(f"Highest Reward Mean = {max_reward:.2f}", xy=(0.5, 0.35), xycoords=xy_cord)
        ax.legend()
        plt.savefig(f'{self.name}/network_fitness_{self.name}.png')
        plt.close()
        # Append data to file
        plot_data = {
            'cur_epoch': [epoch],
            'network_rewards_each_epoch': [self.network_rewards_each_epoch[-1]],
            'network_error_each_epoch': [self.network_error_each_epoch[-1]],
        }
        batch_df = pd.DataFrame(plot_data)
        # Save the batch data to a CSV file
        csv_file = f'./{self.name}/plot_data_{self.name}.csv'
        batch_df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

    def _test_network_single_run(self, repetition):
        state, _ = self.env.reset()
        for _ in range(self.n_trials):
            action, log_action_prob = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)

            if done:
                break
            else:
                state = next_state
        return repetition, self.env.compute_reward()

    def _parallel_test_network(self):
        rep_rewards = np.zeros(self.n_repetitions)

        with ProcessPoolExecutor() as executor:
            future_results = [executor.submit(self._test_network_single_run, repetition) for repetition in
                              range(self.n_repetitions)]

            for future in concurrent.futures.as_completed(future_results):
                repetition, reward = future.result()
                rep_rewards[repetition] = reward

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
        self.save_freq = 20  # How often we save in number of iterations
        self.seed = 1  # Sets the seed of our program, used for reproducibility of results

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            if param != 'name':
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
            'timestamp': [time.time_ns() / 1e9],
        }
        batch_df = pd.DataFrame(epoch_data)
        # Save the batch data to a CSV file
        csv_file = f'./{self.name}/epoch_data_{self.name}.csv'
        batch_df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

        # Reset batch-specific logging data
        self.logger['cur_batch'] = 0
        self.logger['batch_rewards'] = []
        self.logger['actor_losses'] = []