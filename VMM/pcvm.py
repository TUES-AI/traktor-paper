"""PCVM: Path-Conditioned Visual Memory.

PCVM is the lightweight rover-side novelty model for the user's idea:
visual novelty should be conditioned on how the rover got to the current view,
not computed from an isolated frame embedding.

The model keeps a recurrent egocentric state from RGB vision, ultrasonic
distances, IMU yaw rate, previous action, and a rough local dead-reckoned pose.
Novelty is a mix of nearest-cluster distance in that path-conditioned latent,
RND error, and transition surprise.
"""

from collections import deque
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PCVM_CAM_W = 96
PCVM_CAM_H = 64
PCVM_VIS_CHANNELS = 3
PCVM_LATENT_DIM = 128
PCVM_HIDDEN_DIM = 256
PCVM_VIS_FEATURE_DIM = 256
PCVM_MEMORY_SIZE = 1000
PCVM_KNOWN_DIST = 0.015
PCVM_MEMORY_NORM_DIST = 0.05
PCVM_UPDATE_RATE = 0.01
PCVM_VIS_KNOWN_DIST = 0.08
PCVM_VIS_MEMORY_NORM_DIST = 0.16
PCVM_VIS_UPDATE_RATE = 0.01
PCVM_WARMUP_STEPS = 5
PCVM_RND_WEIGHT = 0.15
PCVM_MEMORY_WEIGHT = 0.35
PCVM_VIS_MEMORY_WEIGHT = 0.45
PCVM_SURPRISE_WEIGHT = 0.05
PCVM_YAW_RATE_MAX_DPS = 180.0

PCVM_CANDIDATES_2D = np.array([
    [0.0, 0.25],
    [-0.7, 0.0],
    [0.7, 0.0],
    [-1.0, -1.0],
    [1.0, -1.0],
    [0.0, -0.6],
], dtype=np.float32)
PCVM_CANDIDATES_1D = np.array([[-1.0], [-0.7], [-0.35], [0.0], [0.35], [0.7]], dtype=np.float32)
PCVM_DEFAULT_ACTION_DIM = 1
PCVM_CANDIDATES = PCVM_CANDIDATES_1D


def pcvm_candidates(action_dim=PCVM_DEFAULT_ACTION_DIM):
    return PCVM_CANDIDATES_1D if int(action_dim) == 1 else PCVM_CANDIDATES_2D


def pcvm_tail_dim(action_dim=PCVM_DEFAULT_ACTION_DIM):
    return 3 + 1 + int(action_dim)


def pcvm_obs_dim(action_dim=PCVM_DEFAULT_ACTION_DIM):
    return PCVM_LATENT_DIM + 2 + len(pcvm_candidates(action_dim)) + pcvm_tail_dim(action_dim)


PCVM_OBS_DIM = pcvm_obs_dim(PCVM_DEFAULT_ACTION_DIM)


def clamp(value, lo, hi):
    return max(lo, min(hi, float(value)))


class RunningMean:
    def __init__(self):
        self.n = 0
        self.mean = 0.0

    def ratio(self, value):
        value = float(value)
        self.n += 1
        self.mean += (value - self.mean) / self.n
        return float(np.clip(value / (self.mean + 1e-8), 0.0, 3.0) / 3.0)


class PCVMMemoryBank:
    """Cluster centroids over path-conditioned recurrent latents."""

    def __init__(self, maxlen=PCVM_MEMORY_SIZE, known_dist=PCVM_KNOWN_DIST, update_rate=PCVM_UPDATE_RATE):
        self.bank = []
        self.counts = []
        self.last_seen = []
        self.maxlen = int(maxlen)
        self.known_dist = float(known_dist)
        self.update_rate = float(update_rate)

    def query(self, z):
        if not self.bank:
            return 1.0, None
        bank_t = torch.stack(self.bank)
        sims = (bank_t @ z.T).squeeze(-1)
        best = int(sims.argmax().item())
        return float(1.0 - sims[best].item()), best

    def update(self, z, dist, cluster_idx, step):
        z = z.detach().squeeze(0)
        if cluster_idx is not None and dist < self.known_dist:
            eta = min(self.update_rate, 1.0 / (self.counts[cluster_idx] + 1))
            centroid = F.normalize((1.0 - eta) * self.bank[cluster_idx] + eta * z, dim=0)
            self.bank[cluster_idx] = centroid.detach()
            self.counts[cluster_idx] += 1
            self.last_seen[cluster_idx] = step
            return cluster_idx, False
        if len(self.bank) >= self.maxlen:
            evict = int(np.argmin(self.last_seen))
            self.bank.pop(evict)
            self.counts.pop(evict)
            self.last_seen.pop(evict)
        self.bank.append(z.detach())
        self.counts.append(1)
        self.last_seen.append(step)
        return len(self.bank) - 1, True


class PCVMNet(nn.Module):
    def __init__(self, action_dim=PCVM_DEFAULT_ACTION_DIM):
        super().__init__()
        self.action_dim = int(action_dim)
        self.visual = nn.Sequential(
            nn.Conv2d(PCVM_VIS_CHANNELS, 32, kernel_size=5, stride=2, padding=2), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.GroupNorm(16, 128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), nn.GroupNorm(16, 128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, PCVM_VIS_FEATURE_DIM), nn.ReLU(),
        )
        # sensors(3) + motion(yaw_rate + last_action[action_dim]) + local_pose(x,y,sin,cos)(4)
        self.proprio = nn.Sequential(nn.Linear(8 + self.action_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.gru = nn.GRUCell(PCVM_VIS_FEATURE_DIM + 128, PCVM_HIDDEN_DIM)
        self.proj = nn.Linear(PCVM_HIDDEN_DIM, PCVM_LATENT_DIM)
        self.transition = nn.Sequential(
            nn.Linear(PCVM_LATENT_DIM + self.action_dim, 256), nn.ReLU(), nn.Linear(256, PCVM_LATENT_DIM)
        )
        self.inverse = nn.Sequential(
            nn.Linear(PCVM_LATENT_DIM * 2, 256), nn.ReLU(), nn.Linear(256, self.action_dim), nn.Tanh()
        )
        self.rnd_target = nn.Sequential(nn.Linear(PCVM_LATENT_DIM, 256), nn.ReLU(), nn.Linear(256, 128))
        self.rnd_pred = nn.Sequential(
            nn.Linear(PCVM_LATENT_DIM, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128)
        )
        for p in self.rnd_target.parameters():
            p.requires_grad = False

    def encode(self, visual, proprio, hidden):
        x = torch.cat([self.visual(visual), self.proprio(proprio)], dim=1)
        hidden = self.gru(x, hidden)
        z = F.normalize(self.proj(hidden), dim=1)
        return z, hidden


class PCVM:
    def __init__(self, device=None, action_dim=PCVM_DEFAULT_ACTION_DIM):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.action_dim = int(action_dim)
        self.net = PCVMNet(action_dim=self.action_dim).to(self.device)
        self.opt = torch.optim.Adam(
            list(self.net.visual.parameters())
            + list(self.net.proprio.parameters())
            + list(self.net.gru.parameters())
            + list(self.net.proj.parameters())
            + list(self.net.transition.parameters())
            + list(self.net.inverse.parameters()),
            lr=2e-4,
        )
        self.rnd_opt = torch.optim.Adam(self.net.rnd_pred.parameters(), lr=5e-5)
        self.memory = PCVMMemoryBank()
        self.visual_memory = PCVMMemoryBank(known_dist=PCVM_VIS_KNOWN_DIST, update_rate=PCVM_VIS_UPDATE_RATE)
        self.rnd_norm = RunningMean()
        self.surprise_norm = RunningMean()
        self.losses = deque(maxlen=500)
        self.hidden = torch.zeros(1, PCVM_HIDDEN_DIM, device=self.device)
        self.prev_visual = None
        self.prev_proprio = None
        self.prev_hidden = None
        self.prev_z = None
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.yaw_rad = 0.0
        self.step = 0

    def reset(self):
        self.hidden.zero_()
        self.prev_visual = None
        self.prev_proprio = None
        self.prev_hidden = None
        self.prev_z = None
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.yaw_rad = 0.0

    def preprocess_frame(self, frame_bgr):
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (PCVM_CAM_W, PCVM_CAM_H), interpolation=cv2.INTER_AREA)
        chw = np.transpose(small.astype(np.float32) / 255.0, (2, 0, 1))
        return chw

    def _tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _update_pose(self, action, yaw_rate_norm, dt):
        yaw_rate_dps = clamp(yaw_rate_norm, -1.0, 1.0) * PCVM_YAW_RATE_MAX_DPS
        self.yaw_rad += math.radians(yaw_rate_dps) * max(0.0, float(dt))
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        forward = 1.0 if len(action) == 1 else clamp((float(action[1]) + 1.0) * 0.5, 0.0, 1.0)
        signed = forward if abs(float(action[0])) < 0.9 else 0.25 * forward
        self.pose_x += math.cos(self.yaw_rad) * signed * max(0.0, float(dt))
        self.pose_y += math.sin(self.yaw_rad) * signed * max(0.0, float(dt))
        self.pose_x = clamp(self.pose_x, -10.0, 10.0)
        self.pose_y = clamp(self.pose_y, -10.0, 10.0)

    def _proprio(self, sensors, motion, action, dt):
        yaw_rate_norm = float(motion[0]) if len(motion) else 0.0
        self._update_pose(action, yaw_rate_norm, dt)
        pose = np.array([
            self.pose_x / 10.0,
            self.pose_y / 10.0,
            math.sin(self.yaw_rad),
            math.cos(self.yaw_rad),
        ], dtype=np.float32)
        return np.concatenate([sensors.astype(np.float32), motion.astype(np.float32), pose]).astype(np.float32)

    def _train_transition(self, visual, proprio, action):
        if self.prev_visual is None or self.prev_proprio is None or self.prev_hidden is None or self.prev_z is None:
            return 0.0, None
        z_prev = self.prev_z.detach()
        z_next, _ = self.net.encode(visual, proprio, self.prev_hidden.detach())
        action_t = self._tensor(action)
        pred = self.net.transition(torch.cat([z_prev, action_t], dim=1))
        inv = self.net.inverse(torch.cat([z_prev.detach(), z_next.detach()], dim=1))
        transition_loss = F.mse_loss(pred, z_next.detach())
        inverse_loss = F.mse_loss(inv, action_t)
        loss = transition_loss + 0.2 * inverse_loss
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        raw_surprise = float(transition_loss.detach().item())
        self.losses.append(raw_surprise)
        return self.surprise_norm.ratio(raw_surprise), float(loss.detach().item())

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
        visual_np = self.preprocess_frame(frame_bgr)
        proprio_np = self._proprio(sensors, motion, action, dt)
        visual = self._tensor(visual_np)
        proprio = self._tensor(proprio_np)

        surprise, loss = self._train_transition(visual, proprio, action)
        with torch.no_grad():
            visual_z = F.normalize(self.net.visual(visual), dim=1)
            z, new_hidden = self.net.encode(visual, proprio, self.hidden)
        self.hidden = new_hidden.detach()

        rnd_norm = self._rnd_update(z.detach())
        path_mem_dist, path_cluster_idx = self.memory.query(z.detach())
        visual_mem_dist, visual_cluster_idx = self.visual_memory.query(visual_z.detach())
        if self.step >= PCVM_WARMUP_STEPS:
            path_cluster_idx, path_new_cluster = self.memory.update(z.detach(), path_mem_dist, path_cluster_idx, self.step)
            visual_cluster_idx, visual_new_cluster = self.visual_memory.update(visual_z.detach(), visual_mem_dist, visual_cluster_idx, self.step)
        else:
            path_new_cluster = False
            visual_new_cluster = False
        path_mem_norm = float(np.clip(path_mem_dist / PCVM_MEMORY_NORM_DIST, 0.0, 1.0))
        visual_mem_norm = float(np.clip(visual_mem_dist / PCVM_VIS_MEMORY_NORM_DIST, 0.0, 1.0))
        mem_dist = max(path_mem_dist, visual_mem_dist)
        mem_norm = max(path_mem_norm, visual_mem_norm)
        new_cluster = bool(path_new_cluster or visual_new_cluster)
        novelty = float(np.clip(
            PCVM_MEMORY_WEIGHT * path_mem_norm
            + PCVM_VIS_MEMORY_WEIGHT * visual_mem_norm
            + PCVM_RND_WEIGHT * rnd_norm
            + PCVM_SURPRISE_WEIGHT * surprise,
            0.0,
            1.0,
        ))
        if new_cluster:
            novelty = max(novelty, mem_norm)

        cand = self.candidate_scores(z.detach())
        tail = np.concatenate([sensors, motion]).astype(np.float32)
        obs = np.concatenate([
            z.squeeze(0).detach().cpu().numpy().astype(np.float32),
            np.array([novelty, surprise], dtype=np.float32),
            cand,
            tail,
        ]).astype(np.float32)
        assert obs.shape == (pcvm_obs_dim(self.action_dim),)

        self.prev_visual = visual.detach()
        self.prev_proprio = proprio.detach()
        self.prev_hidden = self.hidden.detach()
        self.prev_z = z.detach()
        self.step += 1

        return {
            'obs': obs,
            'novelty': novelty,
            'pcvm_novelty': novelty,
            'pcvm_surprise': surprise,
            'pcvm_mem_dist': mem_dist,
            'pcvm_mem_norm': mem_norm,
            'pcvm_path_mem_dist': path_mem_dist,
            'pcvm_path_mem_norm': path_mem_norm,
            'pcvm_visual_mem_dist': visual_mem_dist,
            'pcvm_visual_mem_norm': visual_mem_norm,
            'pcvm_rnd_norm': rnd_norm,
            'pcvm_cluster_id': path_cluster_idx,
            'pcvm_path_cluster_id': path_cluster_idx,
            'pcvm_visual_cluster_id': visual_cluster_idx,
            'pcvm_new_cluster': new_cluster,
            'pcvm_path_new_cluster': path_new_cluster,
            'pcvm_visual_new_cluster': visual_new_cluster,
            'pcvm_bank_size': len(self.memory.bank),
            'pcvm_visual_bank_size': len(self.visual_memory.bank),
            'pcvm_loss': loss,
            'pcvm_pose': [self.pose_x, self.pose_y, self.yaw_rad],
            'step': self.step,
        }
