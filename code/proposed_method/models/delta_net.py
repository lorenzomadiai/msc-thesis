"""Neural network architecture for binary switch usefulness prediction."""

import torch
import torch.nn as nn

from common.features import N_FEATURES


class DeltaNet(nn.Module):
    """Simple MLP binary classifier (logit output) for switch usefulness."""

    def __init__(self, input_dim: int = N_FEATURES, hidden_size: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
