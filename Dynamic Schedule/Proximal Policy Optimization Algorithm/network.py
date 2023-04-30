import numpy as np
import torch
from torch import nn

DEVICE = "cpu" if not torch.has_cuda else "cuda:0"


class ForwardNet(nn.Module):
    def __init__(self, input_size: int, hidden_layers: int, hidden_size: int, output_size: int, critic=False):
        """
        Initialize the neural network.

        Args:
            input_size (int): The number of input features.
            hidden_layers (int): The number of hidden layers.
            hidden_size (int): The size of the hidden layers.
            output_size (int): The number of output features.
        """
        super(ForwardNet, self).__init__()
        self.critic = critic
        self.input_layer = nn.Linear(input_size, hidden_size, device=DEVICE)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size, device=DEVICE)
                                            for _ in range(hidden_layers)])
        self.output_layer = nn.Linear(hidden_size, output_size, device=DEVICE)
        self.activation = nn.ReLU()

    @staticmethod
    def _apply_constraints(action_probs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        assignments, trial_numbers = state[:, 9:11], state[:, 13]

        # Create masks for valid actions - CONSTRAINTS
        mask = torch.ones(4).repeat(state.shape[0], 1)
        add_mask = torch.tensor([1e-9, 0, 0, 0]).repeat(state.shape[0], 1)

        # Apply constraints
        mask[(assignments[:, 0] <= trial_numbers - 75) | (assignments[:, 1] <= trial_numbers - 75), 0] = 0
        mask[(assignments[:, 0] >= 25) | (assignments[:, 1] <= trial_numbers - 75), 1] = 0
        # mask[(assignments[:, 0] >= 25), 1] = 0
        mask[(assignments[:, 1] >= 25) | (assignments[:, 0] <= trial_numbers - 75), 2] = 0
        mask[(assignments[:, 0] >= 25) | (assignments[:, 1] >= 25), 3] = 0

        # Apply the mask to the action probabilities
        constrained_probs = ((action_probs + add_mask) * mask)
        constrained_probs /= ((action_probs + add_mask) * mask).sum(dim=1, keepdim=True)

        return constrained_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the neural network.

        Args:
            x (torch.Tensor): The input tensor.
            batch_constraints (bool): Whether to apply constraints in a batch or not.

        Returns:
            torch.Tensor: The output tensor.
        """
        s = False
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=DEVICE, dtype=torch.float)
        if isinstance(x, tuple):
            x = torch.tensor([*x], device=DEVICE, dtype=torch.float)

        out = self.activation(self.input_layer(x))

        for hidden_layer in self.hidden_layers:
            out = self.activation(hidden_layer(out) + out)  # Add skip connection

        out = nn.Softmax(dim=-1)(self.output_layer(out)) if not self.critic else self.output_layer(out)

        # Ensure the input tensor has a batch dimension
        if len(x.shape) == 1:
            s = True
            x = x.unsqueeze(0)

        # Apply constraints if not a critic and batch_constraints is True
        if not self.critic:
            out = self._apply_constraints(out, x)

        return out[0] if s else out
