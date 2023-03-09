import torch
import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):

    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(8, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 4)

    def forward(self, x):
        """
        :param x: input tensor made from:
            1.trial number;
            2, 3. total number of reward allocated in previous trials to alternatives 1 and 2;
            .4. an indication for current trend represented by the values: −1 if trend is not
            applicable (if two previous actions are not identical),0 if the trend is negative,
             and 1 if the trend is positive;
            5. the probability of using the exploration mode (p_explore for the explore mode);
            6. the alternative chosen in last trial (for the inertia mode).
            7. Ratio between number of rewards left to assign and number of trials left
        :return:Probability tensor where:
            1. don't allocate rewards
            2. allocate reward to biased choice
            2. allocate reward to non-biased choice
            2. allocate reward to both choices
        """
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = self.fc3(x)
        return x


def constrain_rewards(x, bias_allocated, non_bias_allocated, trials_left):
    if trials_left == 25 - bias_allocated:
        x[1] = 1
    if trials_left == 25 - non_bias_allocated:
        x[2] = 1
    x[1] = x[1] if bias_allocated < 25 else 0
    x[2] = x[2] if non_bias_allocated < 25 else 0
    x[3] = x[3] if non_bias_allocated < 25 and bias_allocated < 25 else 0
    return x


