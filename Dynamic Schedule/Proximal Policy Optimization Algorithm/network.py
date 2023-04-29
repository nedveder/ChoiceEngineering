import numpy as np
import torch
from torch import nn

DEVICE = "cpu" if not torch.has_cuda else "cuda:0"
HIDDEN_LAYERS = 2


class ForwardNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the neural network.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The size of the hidden layers.
            output_size (int): The number of output features.
        """
        super(ForwardNet, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_size, device=DEVICE)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size, device=DEVICE)
                                            for _ in range(HIDDEN_LAYERS)])
        self.output_layer = nn.Linear(hidden_size, output_size, device=DEVICE)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Convert observation to tensor if it's a numpy array or tuple
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=DEVICE, dtype=torch.float)
        if isinstance(x, tuple):
            x = torch.tensor([*x], device=DEVICE, dtype=torch.float)

        # Forward pass with skip connections
        x = self.activation(self.input_layer(x))

        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x) + x)  # Add skip connection

        out = nn.Softmax(dim=-1)(self.output_layer(x))
        return out
