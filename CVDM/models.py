from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from CVDM.config import CVDMConfig


class ControllableEncoder(nn.Module):
    def __init__(self, config: CVDMConfig):
        super().__init__()
        in_dim = config.dino_dim + config.range_dim + config.action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.phi_dim),
        )

    def forward(self, dino: torch.Tensor, ranges: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([dino, ranges, last_action], dim=-1)
        return F.normalize(self.net(x), dim=-1)


class ForwardDynamics(nn.Module):
    def __init__(self, config: CVDMConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.phi_dim + config.action_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.phi_dim),
        )

    def forward(self, phi: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(torch.cat([phi, action], dim=-1)), dim=-1)


class InverseDynamics(nn.Module):
    def __init__(self, config: CVDMConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.phi_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.action_dim),
            nn.Tanh(),
        )

    def forward(self, phi_t: torch.Tensor, phi_tp1: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([phi_t, phi_tp1], dim=-1))


class RND(nn.Module):
    def __init__(self, config: CVDMConfig):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(config.phi_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.rnd_out_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(config.phi_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.rnd_out_dim),
        )
        for p in self.target.parameters():
            p.requires_grad = False

    def error(self, phi: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y = self.target(phi)
        yhat = self.predictor(phi)
        return F.mse_loss(yhat, y, reduction="none").mean(dim=-1)


class ControllableVisualDynamics(nn.Module):
    def __init__(self, config: CVDMConfig | None = None):
        super().__init__()
        self.config = config or CVDMConfig()
        self.encoder = ControllableEncoder(self.config)
        self.forward_model = ForwardDynamics(self.config)
        self.inverse_model = InverseDynamics(self.config)
        self.rnd = RND(self.config)

    def encode(self, dino: torch.Tensor, ranges: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:
        return self.encoder(dino, ranges, last_action)

    def predict_candidates(self, phi: torch.Tensor, candidate_actions: torch.Tensor) -> torch.Tensor:
        b = phi.shape[0]
        k = candidate_actions.shape[0]
        phi_rep = phi[:, None, :].expand(b, k, -1).reshape(b * k, -1)
        a_rep = candidate_actions[None, :, :].expand(b, k, -1).reshape(b * k, -1)
        pred = self.forward_model(phi_rep, a_rep)
        return pred.reshape(b, k, -1)
