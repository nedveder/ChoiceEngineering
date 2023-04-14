import random
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from CatieAgent import CatieAgent
import matplotlib.pyplot as plt
from IPython.display import clear_output

ALLOCATION_DICT = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}
DEALLOCATION_DICT = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}


class CatieAgentEnv(gym.Env):
    def __init__(self, number_of_trials=100):
        self.agent = CatieAgent(number_of_trials=number_of_trials)
        self.action_space = spaces.MultiBinary(2)
        self.trial_number = 0
        self.max_trials = number_of_trials
        self.assignments = [0, 0]
        self.observation_space = spaces.Tuple([spaces.Discrete(3, start=-1), spaces.Box(0, 1)
                                                  , spaces.Discrete(2), spaces.Box(0, 1), spaces.Box(0, 1)
                                                  , spaces.MultiDiscrete([100, 100]), spaces.Discrete(100)])

    def reset(self, *, seed=None, options=None, ):
        self.agent = CatieAgent()
        self.trial_number = 0
        self.assignments = [0, 0]
        observation = self.agent.get_catie_param(), self.assignments, self.trial_number
        return observation, {}

    def step(self, action):
        choice = self.agent.choose()
        self.agent.receive_outcome(choice, action)
        self.assignments[0] += action[0]
        self.assignments[1] += action[1]
        self.trial_number += 1

        observation = self.agent.get_catie_param(), self.assignments, self.trial_number

        reward = 0
        done = False

        # Check constraints
        if (self.trial_number > 75 and self.assignments[0] < 100 - self.trial_number) \
                or (self.trial_number > 75 and self.assignments[1] < 100 - self.trial_number):
            reward = -100
            done = True
        # Check termination condition
        elif self.trial_number == self.max_trials:
            done = True
            reward = self.compute_reward()

        return observation, reward, done, {}

    def compute_reward(self):
        return self.agent.get_bias()


# Define the recurrent policy network
class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, output_size))

    def forward(self, x):
        out = self.fc(x)
        return out


# Define the agent class
class Agent:
    def __init__(self, env, input_size, hidden_size, output_size, lr):
        self.env = env
        self.policy_net = PolicyNet(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state):
        catie_params, assignments, trial = state
        state_tensor = torch.Tensor(np.concatenate((catie_params, [trial, assignments[0], assignments[1]])))
        action_probs = self.policy_net(state_tensor)
        action_probs = nn.functional.softmax(action_probs, dim=-1)
        # Create a mask for valid actions
        mask = torch.tensor([1.0, 1.0, 1.0, 1.0])
        if assignments[0] >= 25:
            mask[1] = 0
            mask[3] = 0
        if assignments[1] >= 25:
            mask[2] = 0
            mask[3] = 0

        # Apply the mask to the action probabilities
        constrained_probs = action_probs * mask
        constrained_probs = constrained_probs / constrained_probs.sum()
        return constrained_probs

    def train(self, num_episodes):
        plt.figure(figsize=(10, 5))
        episode_rewards = []
        for i in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            action_probs = []
            actions = []
            for _ in range(100):
                action_prob = self.select_action(state)
                action_probs.append(action_prob)
                action = ALLOCATION_DICT[torch.multinomial(action_prob, num_samples=1).item()]
                actions.append(action)

                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                if done:
                    break

                state = next_state

            episode_rewards.append(episode_reward)
            # Calculate the episode return (discounted sum of rewards)
            G = torch.Tensor([episode_reward])

            # Update the policy network
            self.optimizer.zero_grad()

            loss = -G * sum(
                [torch.log(t_probs[DEALLOCATION_DICT[actions[t]]] + 1e-6) for t, t_probs in enumerate(action_probs)])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

            # Update the live plot
            clear_output(wait=True)

            plt.plot(episode_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Rewards')
            plt.show()

            # Print progress
            if i % 10 == 0:
                print('Episode %d: reward = %d' % (i, episode_reward))


if __name__ == '__main__':
    # Define hyper parameters
    torch.autograd.set_detect_anomaly(True, check_nan=True)
    env = CatieAgentEnv()
    torch.random.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    input_size = 8
    hidden_size = 20
    output_size = 4
    lr = 0.001
    num_episodes = 100000

    # Create the environment and agent
    agent = Agent(env, input_size, hidden_size, output_size, lr)

    # Train the agent
    agent.train(num_episodes)

    # Save the trained model
    torch.save(agent.policy_net.state_dict(), 'trained_model.pth')
