"""Relevant MPNet architecture excerpt from https://github.com/ahq1993/MPNet.

Original repo cloned to /tmp/MPNet during summary extraction.
This is a compact reference, not project-ready code.
"""

import torch.nn as nn


class MLP(nn.Module):
    """Pnet-style deep MLP: concat(environment_latent, current_state, goal_state) -> next_state."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 1280), nn.PReLU(), nn.Dropout(),
            nn.Linear(1280, 1024), nn.PReLU(), nn.Dropout(),
            nn.Linear(1024, 896), nn.PReLU(), nn.Dropout(),
            nn.Linear(896, 768), nn.PReLU(), nn.Dropout(),
            nn.Linear(768, 512), nn.PReLU(), nn.Dropout(),
            nn.Linear(512, 384), nn.PReLU(), nn.Dropout(),
            nn.Linear(384, 256), nn.PReLU(), nn.Dropout(),
            nn.Linear(256, 256), nn.PReLU(), nn.Dropout(),
            nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
            nn.Linear(128, 64), nn.PReLU(), nn.Dropout(),
            nn.Linear(64, 32), nn.PReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        return self.fc(x)


class Encoder(nn.Module):
    """Contractive-autoencoder obstacle encoder used in the simple 2D reference code."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2800, 512), nn.PReLU(),
            nn.Linear(512, 256), nn.PReLU(),
            nn.Linear(256, 128), nn.PReLU(),
            nn.Linear(128, 28),
        )

    def forward(self, x):
        return self.encoder(x)
