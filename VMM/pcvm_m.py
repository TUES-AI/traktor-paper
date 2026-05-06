"""PCVM-M: Path-Conditioned Visual Memory with MobileNet visual features.

This is a diagnostic version of PCVM. It borrows only the strong ImageNet
MobileNetV3 visual encoder idea from VMM, then keeps the user's path-conditioned
GRU, sensors, IMU/action context, local pose estimate, memory bank, RND, and
transition surprise. It does not use VMM temporal smoothing, VMM memory, or VMM
novelty decisions.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from VMM.pcvm import (
    PCVM_CANDIDATES,
    PCVM_HIDDEN_DIM,
    PCVM_KNOWN_DIST,
    PCVM_LATENT_DIM,
    PCVM_MEMORY_NORM_DIST,
    PCVM_OBS_DIM,
    PCVM_RND_WEIGHT,
    PCVM_SURPRISE_WEIGHT,
    PCVM_MEMORY_WEIGHT,
    PCVM_WARMUP_STEPS,
    PCVM_YAW_RATE_MAX_DPS,
    PCVMMemoryBank,
    RunningMean,
    clamp,
)


class MobileNetVisualEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        import torchvision.models as models

        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(576, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim), nn.ReLU())
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            x = self.pool(x).flatten(1)
        return self.proj(x)


class PCVMmNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = MobileNetVisualEncoder(out_dim=256)
        self.proprio = nn.Sequential(nn.Linear(11, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.gru = nn.GRUCell(256 + 128, PCVM_HIDDEN_DIM)
        self.proj = nn.Linear(PCVM_HIDDEN_DIM, PCVM_LATENT_DIM)
        self.transition = nn.Sequential(
            nn.Linear(PCVM_LATENT_DIM + 2, 256), nn.ReLU(), nn.Linear(256, PCVM_LATENT_DIM)
        )
        self.inverse = nn.Sequential(
            nn.Linear(PCVM_LATENT_DIM * 2, 256), nn.ReLU(), nn.Linear(256, 2), nn.Tanh()
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


class PCVMMobileNet:
    def __init__(self, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.net = PCVMmNet().to(self.device)
        self.opt = torch.optim.Adam(
            list(self.net.visual.proj.parameters())
            + list(self.net.proprio.parameters())
            + list(self.net.gru.parameters())
            + list(self.net.proj.parameters())
            + list(self.net.transition.parameters())
            + list(self.net.inverse.parameters()),
            lr=2e-4,
        )
        self.rnd_opt = torch.optim.Adam(self.net.rnd_pred.parameters(), lr=5e-5)
        self.memory = PCVMMemoryBank()
        self.rnd_norm = RunningMean()
        self.surprise_norm = RunningMean()
        self.hidden = torch.zeros(1, PCVM_HIDDEN_DIM, device=self.device)
        self.prev_visual = None
        self.prev_proprio = None
        self.prev_hidden = None
        self.prev_z = None
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.yaw_rad = 0.0
        self.step = 0

    def preprocess_frame(self, frame_bgr):
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_AREA)
        x = small.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        return np.transpose(x, (2, 0, 1)).astype(np.float32)

    def _tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _update_pose(self, action, yaw_rate_norm, dt):
        yaw_rate_dps = clamp(yaw_rate_norm, -1.0, 1.0) * PCVM_YAW_RATE_MAX_DPS
        self.yaw_rad += math.radians(yaw_rate_dps) * max(0.0, float(dt))
        forward = clamp((float(action[1]) + 1.0) * 0.5, 0.0, 1.0)
        signed = forward if abs(float(action[0])) < 0.9 else 0.25 * forward
        self.pose_x = clamp(self.pose_x + math.cos(self.yaw_rad) * signed * max(0.0, float(dt)), -10.0, 10.0)
        self.pose_y = clamp(self.pose_y + math.sin(self.yaw_rad) * signed * max(0.0, float(dt)), -10.0, 10.0)

    def _proprio(self, sensors, motion, action, dt):
        yaw_rate_norm = float(motion[0]) if len(motion) else 0.0
        self._update_pose(action, yaw_rate_norm, dt)
        pose = np.array([self.pose_x / 10.0, self.pose_y / 10.0, math.sin(self.yaw_rad), math.cos(self.yaw_rad)], dtype=np.float32)
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
        loss = transition_loss + 0.2 * F.mse_loss(inv, action_t)
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
        with torch.no_grad():
            zt = z.repeat(len(PCVM_CANDIDATES), 1)
            at = torch.as_tensor(PCVM_CANDIDATES, dtype=torch.float32, device=self.device)
            pred = F.normalize(self.net.transition(torch.cat([zt, at], dim=1)), dim=1)
            score = torch.linalg.vector_norm(pred - zt, dim=1)
            score = score / (score.mean() + 1e-6)
            return torch.clamp(score / 3.0, 0, 1).detach().cpu().numpy().astype(np.float32)

    def observe(self, frame_bgr, sensors, motion, action, dt):
        action = np.asarray(action, dtype=np.float32)
        sensors = np.asarray(sensors, dtype=np.float32)
        motion = np.asarray(motion, dtype=np.float32)
        visual = self._tensor(self.preprocess_frame(frame_bgr))
        proprio = self._tensor(self._proprio(sensors, motion, action, dt))

        surprise, loss = self._train_transition(visual, proprio, action)
        with torch.no_grad():
            z, new_hidden = self.net.encode(visual, proprio, self.hidden)
        self.hidden = new_hidden.detach()

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
        assert obs.shape == (PCVM_OBS_DIM,)

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
            'pcvm_rnd_norm': rnd_norm,
            'pcvm_cluster_id': cluster_idx,
            'pcvm_new_cluster': new_cluster,
            'pcvm_bank_size': len(self.memory.bank),
            'pcvm_loss': loss,
            'pcvm_pose': [self.pose_x, self.pose_y, self.yaw_rad],
            'pcvm_visual': 'mobilenet_v3_small_imagenet_no_vmm_smoothing',
            'step': self.step,
        }
