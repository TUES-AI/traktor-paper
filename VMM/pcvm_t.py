"""PCVM-T: transformer path-conditioned visual memory.

This backend keeps the same SAC observation contract as PCVM but replaces the
single-step recurrent context with a temporal transformer over recent visual +
proprioceptive tokens. It is intentionally separate from PCVM/PCVM-M so rover
tests can compare attention against the current CNN/GRU variants.
"""

from collections import deque
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from VMM.pcvm import (
    PCVM_CAM_H,
    PCVM_CAM_W,
    PCVM_CANDIDATES,
    PCVM_DEFAULT_ACTION_DIM,
    PCVM_KNOWN_DIST,
    PCVM_LATENT_DIM,
    PCVM_MEMORY_NORM_DIST,
    PCVM_MEMORY_WEIGHT,
    PCVM_OBS_DIM,
    PCVM_RND_WEIGHT,
    PCVM_SURPRISE_WEIGHT,
    PCVM_VIS_CHANNELS,
    PCVM_WARMUP_STEPS,
    PCVM_YAW_RATE_MAX_DPS,
    PCVMMemoryBank,
    RunningMean,
    clamp,
    pcvm_candidates,
    pcvm_obs_dim,
)


PCVMT_TOKEN_DIM = 256
PCVMT_CONTEXT_LEN = 16
PCVMT_NHEAD = 8
PCVMT_LAYERS = 4


class BigVisualCNN(nn.Module):
    def __init__(self, out_dim=PCVMT_TOKEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(PCVM_VIS_CHANNELS, 64, kernel_size=5, stride=2, padding=2), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.GroupNorm(16, 128), nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.GroupNorm(16, 256), nn.SiLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), nn.GroupNorm(16, 256), nn.SiLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.GroupNorm(16, 256), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class PCVMTransformerNet(nn.Module):
    def __init__(self, action_dim=PCVM_DEFAULT_ACTION_DIM):
        super().__init__()
        self.action_dim = int(action_dim)
        self.visual = BigVisualCNN(PCVMT_TOKEN_DIM)
        self.proprio = nn.Sequential(nn.Linear(8 + self.action_dim, 128), nn.SiLU(), nn.Linear(128, PCVMT_TOKEN_DIM), nn.SiLU())
        self.token_proj = nn.Sequential(nn.Linear(PCVMT_TOKEN_DIM * 2, PCVMT_TOKEN_DIM), nn.SiLU())
        self.cls = nn.Parameter(torch.zeros(1, 1, PCVMT_TOKEN_DIM))
        self.pos = nn.Parameter(torch.randn(1, PCVMT_CONTEXT_LEN + 1, PCVMT_TOKEN_DIM) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=PCVMT_TOKEN_DIM,
            nhead=PCVMT_NHEAD,
            dim_feedforward=PCVMT_TOKEN_DIM * 4,
            dropout=0.05,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=PCVMT_LAYERS)
        self.proj = nn.Sequential(nn.LayerNorm(PCVMT_TOKEN_DIM), nn.Linear(PCVMT_TOKEN_DIM, PCVM_LATENT_DIM))
        self.transition = nn.Sequential(
            nn.Linear(PCVM_LATENT_DIM + self.action_dim, 512), nn.SiLU(), nn.Linear(512, PCVM_LATENT_DIM)
        )
        self.inverse = nn.Sequential(
            nn.Linear(PCVM_LATENT_DIM * 2, 512), nn.SiLU(), nn.Linear(512, self.action_dim), nn.Tanh()
        )
        self.rnd_target = nn.Sequential(nn.Linear(PCVM_LATENT_DIM, 512), nn.SiLU(), nn.Linear(512, 128))
        self.rnd_pred = nn.Sequential(
            nn.Linear(PCVM_LATENT_DIM, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 128)
        )
        for p in self.rnd_target.parameters():
            p.requires_grad = False

    def make_token(self, visual, proprio):
        return self.token_proj(torch.cat([self.visual(visual), self.proprio(proprio)], dim=1))

    def encode_tokens(self, tokens):
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        b = tokens.shape[0]
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        x = x + self.pos[:, :x.shape[1], :]
        x = self.transformer(x)
        return F.normalize(self.proj(x[:, 0]), dim=1)


class PCVMTransformer:
    def __init__(self, device=None, action_dim=PCVM_DEFAULT_ACTION_DIM):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.action_dim = int(action_dim)
        self.net = PCVMTransformerNet(action_dim=self.action_dim).to(self.device)
        self.opt = torch.optim.Adam(
            list(self.net.visual.parameters())
            + list(self.net.proprio.parameters())
            + list(self.net.token_proj.parameters())
            + list(self.net.transformer.parameters())
            + list(self.net.proj.parameters())
            + list(self.net.transition.parameters())
            + list(self.net.inverse.parameters()),
            lr=2e-4,
        )
        self.rnd_opt = torch.optim.Adam(self.net.rnd_pred.parameters(), lr=5e-5)
        self.memory = PCVMMemoryBank()
        self.rnd_norm = RunningMean()
        self.surprise_norm = RunningMean()
        self.tokens = deque(maxlen=PCVMT_CONTEXT_LEN)
        self.prev_z = None
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.yaw_rad = 0.0
        self.step = 0

    def preprocess_frame(self, frame_bgr):
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (PCVM_CAM_W, PCVM_CAM_H), interpolation=cv2.INTER_AREA)
        return np.transpose(small.astype(np.float32) / 255.0, (2, 0, 1)).astype(np.float32)

    def _tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _update_pose(self, action, yaw_rate_norm, dt):
        yaw_rate_dps = clamp(yaw_rate_norm, -1.0, 1.0) * PCVM_YAW_RATE_MAX_DPS
        self.yaw_rad += math.radians(yaw_rate_dps) * max(0.0, float(dt))
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        forward = 1.0 if len(action) == 1 else clamp((float(action[1]) + 1.0) * 0.5, 0.0, 1.0)
        signed = forward if abs(float(action[0])) < 0.9 else 0.25 * forward
        self.pose_x = clamp(self.pose_x + math.cos(self.yaw_rad) * signed * max(0.0, float(dt)), -10.0, 10.0)
        self.pose_y = clamp(self.pose_y + math.sin(self.yaw_rad) * signed * max(0.0, float(dt)), -10.0, 10.0)

    def _proprio(self, sensors, motion, action, dt):
        yaw_rate_norm = float(motion[0]) if len(motion) else 0.0
        self._update_pose(action, yaw_rate_norm, dt)
        pose = np.array([self.pose_x / 10.0, self.pose_y / 10.0, math.sin(self.yaw_rad), math.cos(self.yaw_rad)], dtype=np.float32)
        return np.concatenate([sensors.astype(np.float32), motion.astype(np.float32), pose]).astype(np.float32)

    def _token_sequence(self, token):
        old = list(self.tokens)[-(PCVMT_CONTEXT_LEN - 1):]
        pad_n = max(0, PCVMT_CONTEXT_LEN - len(old) - 1)
        pad = [torch.zeros_like(token.squeeze(0)) for _ in range(pad_n)]
        seq = pad + old + [token.squeeze(0)]
        return torch.stack(seq, dim=0).unsqueeze(0)

    def _train_transition(self, z, action):
        if self.prev_z is None:
            return 0.0, None
        action_t = self._tensor(action)
        pred = self.net.transition(torch.cat([self.prev_z.detach(), action_t], dim=1))
        inv = self.net.inverse(torch.cat([self.prev_z.detach(), z.detach()], dim=1))
        transition_loss = F.mse_loss(pred, z.detach())
        inverse_loss = F.mse_loss(inv, action_t)
        loss = transition_loss + 0.2 * inverse_loss
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return self.surprise_norm.ratio(float(transition_loss.detach().item())), float(loss.detach().item())

    def _rnd_update(self, z):
        with torch.no_grad():
            target = self.net.rnd_target(z)
        pred = self.net.rnd_pred(z)
        loss = F.mse_loss(pred, target)
        self.rnd_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.rnd_opt.step()
        return self.rnd_norm.ratio(float(loss.detach().item()))

    def candidate_scores(self, z):
        candidates = pcvm_candidates(self.action_dim)
        with torch.no_grad():
            zt = z.repeat(len(candidates), 1)
            at = torch.as_tensor(candidates, dtype=torch.float32, device=self.device)
            pred = F.normalize(self.net.transition(torch.cat([zt, at], dim=1)), dim=1)
            score = torch.linalg.vector_norm(pred - zt, dim=1)
            score = score / (score.mean() + 1e-6)
            return torch.clamp(score / 3.0, 0, 1).detach().cpu().numpy().astype(np.float32)

    def observe(self, frame_bgr, sensors, motion, action, dt):
        action = np.asarray(action, dtype=np.float32).reshape(-1)[:self.action_dim]
        if len(action) < self.action_dim:
            action = np.pad(action, (0, self.action_dim - len(action))).astype(np.float32)
        sensors = np.asarray(sensors, dtype=np.float32)
        motion = np.asarray(motion, dtype=np.float32)
        visual = self._tensor(self.preprocess_frame(frame_bgr))
        proprio = self._tensor(self._proprio(sensors, motion, action, dt))
        token = self.net.make_token(visual, proprio)
        seq = self._token_sequence(token)

        with torch.no_grad():
            z = self.net.encode_tokens(seq)
        surprise, loss = self._train_transition(z, action)
        rnd_norm = self._rnd_update(z.detach())

        mem_dist, cluster_idx = self.memory.query(z.detach())
        if self.step >= PCVM_WARMUP_STEPS:
            cluster_idx, new_cluster = self.memory.update(z.detach(), mem_dist, cluster_idx, self.step)
        else:
            new_cluster = False
        mem_norm = float(np.clip(mem_dist / PCVM_MEMORY_NORM_DIST, 0.0, 1.0))
        novelty = float(np.clip(
            PCVM_MEMORY_WEIGHT * mem_norm + PCVM_RND_WEIGHT * rnd_norm + PCVM_SURPRISE_WEIGHT * surprise,
            0.0,
            1.0,
        ))
        if new_cluster:
            novelty = max(novelty, mem_norm)

        obs = np.concatenate([
            z.squeeze(0).detach().cpu().numpy().astype(np.float32),
            np.array([novelty, surprise], dtype=np.float32),
            self.candidate_scores(z.detach()),
            np.concatenate([sensors, motion]).astype(np.float32),
        ]).astype(np.float32)
        assert obs.shape == (pcvm_obs_dim(self.action_dim),)

        self.tokens.append(token.detach().squeeze(0))
        self.prev_z = z.detach()
        self.step += 1
        return {
            'obs': obs,
            'novelty': novelty,
            'pcvm_novelty': novelty,
            'pcvm_surprise': surprise,
            'pcvm_mem_dist': mem_dist,
            'pcvm_mem_norm': mem_norm,
            'pcvm_rnd_norm': rnd_norm,
            'pcvm_cluster_id': cluster_idx,
            'pcvm_new_cluster': new_cluster,
            'pcvm_bank_size': len(self.memory.bank),
            'pcvm_loss': loss,
            'pcvm_pose': [self.pose_x, self.pose_y, self.yaw_rad],
            'pcvm_visual': 'big_cnn_temporal_transformer',
            'pcvm_context_len': len(self.tokens),
            'step': self.step,
        }
