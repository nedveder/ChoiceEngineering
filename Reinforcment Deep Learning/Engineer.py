import random

import gymnasium as gym
from Environment import CatieAgentEnv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ALLOCATION_DICT = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}
DEALLOCATION_DICT = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}


# Define the recurrent policy network
class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


# Define the agent class
class Agent:
    def __init__(self, env, input_size, hidden_size, output_size, lr, gamma):
        self.env = env
        self.policy_net = PolicyNet(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.episode_rewards = []

    def select_action(self, state):
        catie_params, assignments, trial = state
        state_tensor = torch.Tensor(np.concatenate((catie_params, [trial, assignments[0], assignments[1]])))
        action_probs = self.policy_net(state_tensor)
        action_probs = nn.functional.softmax(action_probs, dim=-1)

        constrained_probs = action_probs.clone()
        # Apply constraints
        if assignments[0] >= 25:
            constrained_probs[1].zero_()
            constrained_probs[3].zero_()
        if assignments[1] >= 25:
            constrained_probs[2].zero_()
            constrained_probs[3].zero_()

        constrained_probs = constrained_probs / constrained_probs.sum()
        return constrained_probs

    def train(self, num_episodes, max_steps):
        for i in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            action_probs = []
            actions = []
            for t in range(max_steps):
                action_prob = self.select_action(state)
                action_probs.append(action_prob)
                action = ALLOCATION_DICT[torch.multinomial(action_prob, num_samples=1).item()]
                actions.append(action)

                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                if done or t == max_steps - 1:
                    self.episode_rewards.append(episode_reward)
                    break

                state = next_state

            # Calculate the episode return (discounted sum of rewards)
            G = torch.Tensor([sum(self.episode_rewards) * (self.gamma ** len(self.episode_rewards))])

            # Update the policy network
            self.optimizer.zero_grad()
            for t, t_probs in enumerate(action_probs):
                loss = -G * torch.clamp(t_probs[DEALLOCATION_DICT[actions[t]]], min=1e-6).log()
                loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

            # Print progress
            if i % 10 == 0:
                print('Episode %d: reward = %d' % (i, episode_reward))


if __name__ == '__main__':
    # Define hyperparameters
    torch.autograd.set_detect_anomaly(True, check_nan=True)
    env = CatieAgentEnv()
    torch.random.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    input_size = 8
    hidden_size = 20
    output_size = 4
    lr = 0.005
    gamma = 1.5
    num_episodes = 100000
    max_steps = 101

    # Create the environment and agent
    agent = Agent(env, input_size, hidden_size, output_size, lr, gamma)

    # Train the agent
    agent.train(num_episodes, max_steps)
    input(agent.policy_net.parameters())
