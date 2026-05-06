from __future__ import annotations

from collections import deque
import itertools

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RunningMean:
    def __init__(self):
        self.n = 0
        self.mean = 0.0

    def update_ratio(self, x: float) -> float:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        return float(np.clip(x / (self.mean + 1e-8), 0.0, 3.0) / 3.0)


class RND(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 64, lr: float = 1e-4):
        super().__init__()
        self.target = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)).to(DEVICE)
        self.pred = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)).to(DEVICE)
        for p in self.target.parameters():
            p.requires_grad = False
        self.opt = torch.optim.Adam(self.pred.parameters(), lr=lr)
        self.norm = RunningMean()

    def score_update(self, x_np: np.ndarray) -> float:
        x = torch.as_tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            t = self.target(x)
        p = self.pred(x)
        loss = F.mse_loss(p, t)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return self.norm.update_ratio(float(loss.item()))

    @torch.no_grad()
    def score(self, x_np: np.ndarray) -> float:
        x = torch.as_tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        return self.norm.update_ratio(float(F.mse_loss(self.pred(x), self.target(x)).item()))


class SACVMMWrapper(gym.Wrapper):
    """Current-state visual/sensor RND without privileged pose or coverage."""

    def __init__(self, env: gym.Env, novelty_scale: float = 0.8, warmup: int = 200):
        super().__init__(env)
        self.rnd = RND(int(np.prod(env.observation_space.shape)), lr=5e-5)
        self.novelty_scale = novelty_scale
        self.warmup = warmup
        self.steps = 0
        low = np.concatenate([env.observation_space.low, np.array([0.0], dtype=np.float32)])
        high = np.concatenate([env.observation_space.high, np.array([1.0], dtype=np.float32)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self._nov = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._nov = 0.0
        return self._augment(obs), info

    def _augment(self, obs):
        return np.concatenate([obs.astype(np.float32), np.array([self._nov], dtype=np.float32)])

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.steps += 1
        self._nov = self.rnd.score_update(obs)
        if self.steps >= self.warmup:
            reward += self.novelty_scale * self._nov
        info["vmm_novelty"] = self._nov
        return self._augment(obs), reward, term, trunc, info


class PredictiveStateModel(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 2, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, latent_dim), nn.Tanh())
        self.transition = nn.Sequential(nn.Linear(latent_dim + action_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim))
        self.inverse = nn.Sequential(nn.Linear(latent_dim * 2, 128), nn.ReLU(), nn.Linear(128, action_dim), nn.Tanh())
        self.rnd = RND(latent_dim, hidden=128, out_dim=64, lr=5e-5)

    def encode_np(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            return self.encoder(x).squeeze(0).detach().cpu().numpy().astype(np.float32)


class PredictiveVMMWrapper(gym.Wrapper):
    """Online action-conditioned latent model used as the policy observation.

    This keeps the system one-family: an observation encoder, transition model,
    inverse-action consistency head, RND over the learned latent, and candidate
    transition-surprise estimates for action selection.
    """

    CANDIDATES = np.array([
        [0.0, 0.8],   # forward
        [-0.7, 0.55], # arc/right-ish
        [0.7, 0.55],  # arc/left-ish
        [-1.0, 0.25], # spin right
        [1.0, 0.25],  # spin left
        [0.0, -0.45], # reverse
    ], dtype=np.float32)

    def __init__(self, env: gym.Env, novelty_scale: float = 0.9, surprise_scale: float = 0.25, warmup: int = 200):
        super().__init__(env)
        self.raw_dim = int(np.prod(env.observation_space.shape))
        self.model = PredictiveStateModel(self.raw_dim).to(DEVICE)
        self.opt = torch.optim.Adam(itertools.chain(
            self.model.encoder.parameters(), self.model.transition.parameters(), self.model.inverse.parameters()), lr=2e-4)
        self.latent_dim = 64
        self.novelty_scale = novelty_scale
        self.surprise_scale = surprise_scale
        self.warmup = warmup
        self.steps = 0
        self.prev_raw = None
        self.prev_action = None
        self.nov = 0.0
        self.surprise = 0.0
        self.losses = deque(maxlen=500)
        # latent + current novelty/surprise + candidate predicted change + original 7 motion/sensor scalars
        dim = self.latent_dim + 2 + len(self.CANDIDATES) + 7
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)

    def reset(self, **kwargs):
        raw, info = self.env.reset(**kwargs)
        self.prev_raw = raw.copy()
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.nov = self.surprise = 0.0
        return self._obs(raw), info

    def _candidate_scores(self, z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            zt = torch.as_tensor(z, dtype=torch.float32, device=DEVICE).unsqueeze(0).repeat(len(self.CANDIDATES), 1)
            at = torch.as_tensor(self.CANDIDATES, dtype=torch.float32, device=DEVICE)
            pred = self.model.transition(torch.cat([zt, at], dim=1))
            score = torch.linalg.vector_norm(pred - zt, dim=1)
            score = score / (score.mean() + 1e-6)
            return torch.clamp(score / 3.0, 0, 1).detach().cpu().numpy().astype(np.float32)

    def _obs(self, raw: np.ndarray) -> np.ndarray:
        z = self.model.encode_np(raw)
        cand = self._candidate_scores(z)
        tail = raw[-7:].astype(np.float32) * 2.0 - 1.0
        return np.concatenate([z, np.array([self.nov, self.surprise], dtype=np.float32), cand, tail]).astype(np.float32)

    def _train_transition(self, raw: np.ndarray, action: np.ndarray, next_raw: np.ndarray) -> float:
        x = torch.as_tensor(raw, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        xp = torch.as_tensor(next_raw, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        a = torch.as_tensor(action, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        z = self.model.encoder(x)
        zp = self.model.encoder(xp)
        pred = self.model.transition(torch.cat([z, a], dim=1))
        inv = self.model.inverse(torch.cat([z.detach(), zp.detach()], dim=1))
        transition_loss = F.mse_loss(pred, zp.detach())
        inverse_loss = F.mse_loss(inv, a)
        # Small temporal smoothness keeps the latent from becoming pure pixel hash.
        smooth_loss = 0.02 * torch.linalg.vector_norm(zp - z, dim=1).mean()
        loss = transition_loss + 0.2 * inverse_loss + smooth_loss
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        raw_surprise = float(transition_loss.detach().item())
        self.losses.append(raw_surprise)
        mean = float(np.mean(self.losses)) + 1e-8
        self.surprise = float(np.clip(raw_surprise / mean, 0, 3) / 3.0)
        return float(loss.detach().item())

    def step(self, action):
        raw, reward, term, trunc, info = self.env.step(action)
        self.steps += 1
        if self.prev_raw is not None:
            loss = self._train_transition(self.prev_raw, np.asarray(action, dtype=np.float32), raw)
            info["pvmm_loss"] = loss
        z = self.model.encode_np(raw)
        self.nov = self.model.rnd.score_update(z)
        if self.steps >= self.warmup:
            reward += self.novelty_scale * self.nov + self.surprise_scale * self.surprise
        info["pvmm_novelty"] = self.nov
        info["pvmm_surprise"] = self.surprise
        self.prev_raw = raw.copy()
        self.prev_action = np.asarray(action, dtype=np.float32).copy()
        return self._obs(raw), reward, term, trunc, info
