"""Tiny sensory recurrent model for mapless world-feedback exploration.

The runtime policy receives only compact onboard feedback:
- ultrasonic range readings
- IMU/motion summaries
- previous high-level action
- executor outcome from the last action

There are no pixels, no visual memory bank, and no visual clustering in this
module. Offline DINO clustering lives under tools/ and is not imported here.
"""

from dataclasses import dataclass

import torch
from torch import nn


RANGE_DIM = 3
MOTION_DIM = 4
ACTION_DIM = 1
EXECUTOR_DIM = 4
TWF_INPUT_DIM = RANGE_DIM + MOTION_DIM + ACTION_DIM + EXECUTOR_DIM
TWF_OBS_DIM = 64


@dataclass(frozen=True)
class TWFPolicyInput:
    """Shape contract for one policy step."""

    ranges: tuple[float, float, float]
    motion: tuple[float, float, float, float]
    previous_action: float
    executor_feedback: tuple[float, float, float, float]


class TWFEncoder(nn.Module):
    """Small MLP + GRU encoder for SAC observations."""

    def __init__(self, input_dim=TWF_INPUT_DIM, hidden_dim=TWF_OBS_DIM):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.sensor_mlp = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.gru = nn.GRUCell(64, self.hidden_dim)
        self.out = nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Tanh())

    def initial_state(self, batch_size=1, device=None):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, x, hidden=None):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if hidden is None:
            hidden = self.initial_state(x.shape[0], x.device)
        y = self.sensor_mlp(x.float())
        hidden = self.gru(y, hidden)
        return self.out(hidden), hidden
