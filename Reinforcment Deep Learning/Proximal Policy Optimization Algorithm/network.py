import numpy as np
import torch
from torch import nn

DEVICE = "cpu" if not torch.has_cuda else "cuda:0"
HIDDEN_LAYERS = 2


class ForwardNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the PolicyNet neural network.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The size of the hidden layers.
            output_size (int): The number of output features.
            activation_function (callable): The activation function to be used in the hidden layers.
        """
        super(ForwardNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, hidden_size),
                                nn.ReLU(),
                                *([nn.LazyLinear(hidden_size, device=DEVICE), nn.ReLU()] * HIDDEN_LAYERS),
                                nn.Linear(hidden_size, output_size),
                                nn.Softmax(dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the PolicyNet neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Convert observation to tensor if it's a numpy array
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        if isinstance(x, tuple):
            x = torch.tensor([*x], dtype=torch.float)
        out = self.fc(x)
        return out
