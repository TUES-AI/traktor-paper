"""PCVM-D3: PCVM with frozen quantized DINOv3 ONNX visual features.

This backend is the real-rover candidate after offline replay calibration:
- encoder: onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX
- variant: model_quantized
- input: 336x336
- visual memory threshold: 1.0
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from VMM.pcvm import (
    PCVM_DEFAULT_ACTION_DIM,
    PCVM_HIDDEN_DIM,
    PCVM_KNOWN_DIST,
    PCVM_LATENT_DIM,
    PCVM_MEMORY_NORM_DIST,
    PCVM_MEMORY_WEIGHT,
    PCVM_RND_WEIGHT,
    PCVM_SURPRISE_WEIGHT,
    PCVM_VIS_MEMORY_WEIGHT,
    PCVM_VIS_UPDATE_RATE,
    PCVM_WARMUP_STEPS,
    PCVM_YAW_RATE_MAX_DPS,
    PCVMMemoryBank,
    RunningMean,
    clamp,
    pcvm_candidates,
    pcvm_obs_dim,
)


DINO3_ONNX_REPO = 'onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX'
DINO3_ONNX_VARIANT = 'model_quantized'
DINO3_INPUT_SIZE = 336
DINO3_FEATURE_DIM = 384
DINO3_VIS_KNOWN_DIST = 1.00
DINO3_VIS_MEMORY_NORM_DIST = 2.20


class DINO3L2MemoryBank(PCVMMemoryBank):
    """Memory bank using the same normalized L2 distance as the DINOv3 replay tool."""

    def query(self, z):
        z = z.detach().squeeze(0)
        if not self.bank:
            return 1.0, None
        bank_t = torch.stack(self.bank).to(z.device)
        dists = torch.linalg.vector_norm(bank_t - z.unsqueeze(0), dim=1)
        best = int(dists.argmin().item())
        return float(dists[best].item()), best


class DINOv3ONNXVisualEncoder:
    def __init__(self, repo=DINO3_ONNX_REPO, variant=DINO3_ONNX_VARIANT, input_size=DINO3_INPUT_SIZE, threads=4):
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort

        self.repo = repo
        self.variant = variant
        self.input_size = int(input_size)
        hf_hub_download(repo, f'onnx/{variant}.onnx_data')
        model_path = hf_hub_download(repo, f'onnx/{variant}.onnx')
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(threads)
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess_frame(self, frame_bgr):
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        return np.transpose(x, (2, 0, 1))[None].astype(np.float32)

    def encode(self, frame_bgr):
        x = self.preprocess_frame(frame_bgr)
        outputs = self.session.run(None, {self.input_name: x})
        return outputs[1].astype(np.float32)


class PCVMD3Net(nn.Module):
    def __init__(self, action_dim=PCVM_DEFAULT_ACTION_DIM):
        super().__init__()
        self.action_dim = int(action_dim)
        self.proprio = nn.Sequential(nn.Linear(8 + self.action_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.gru = nn.GRUCell(DINO3_FEATURE_DIM + 128, PCVM_HIDDEN_DIM)
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

    def encode(self, visual_feat, proprio, hidden):
        visual = F.normalize(visual_feat, dim=1)
        prop = self.proprio(proprio)
        hidden = self.gru(torch.cat([visual, prop], dim=1), hidden)
        z = F.normalize(self.proj(hidden), dim=1)
        return z, hidden

    def visual_embedding(self, visual_feat):
        return F.normalize(visual_feat, dim=1)


class PCVMDINOv3ONNX:
    def __init__(self, device=None, action_dim=PCVM_DEFAULT_ACTION_DIM, input_size=DINO3_INPUT_SIZE, threads=4):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.action_dim = int(action_dim)
        self.visual_encoder = DINOv3ONNXVisualEncoder(input_size=input_size, threads=threads)
        self.net = PCVMD3Net(action_dim=self.action_dim).to(self.device)
        self.opt = torch.optim.Adam(
            list(self.net.proprio.parameters())
            + list(self.net.gru.parameters())
            + list(self.net.proj.parameters())
            + list(self.net.transition.parameters())
            + list(self.net.inverse.parameters()),
            lr=2e-4,
        )
        self.rnd_opt = torch.optim.Adam(self.net.rnd_pred.parameters(), lr=5e-5)
        # PCVM-D3 intentionally does not use recurrent/path clusters for reward.
        # RND/surprise still operate on the recurrent latent, but the explicit
        # memory-bank novelty is visual-only to avoid rewarding hidden-state or
        # pose/action drift during loops.
        self.memory = None
        self.visual_memory = DINO3L2MemoryBank(known_dist=DINO3_VIS_KNOWN_DIST, update_rate=PCVM_VIS_UPDATE_RATE)
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

    def reset(self):
        self.hidden.zero_()
        self.prev_visual = None
        self.prev_proprio = None
        self.prev_hidden = None
        self.prev_z = None
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.yaw_rad = 0.0

    def _tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

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

    def _train_transition(self, visual_feat, proprio, action):
        if self.prev_visual is None or self.prev_proprio is None or self.prev_hidden is None or self.prev_z is None:
            return 0.0, None
        z_prev = self.prev_z.detach()
        z_next, _ = self.net.encode(visual_feat, proprio, self.prev_hidden.detach())
        action_t = self._tensor(action).unsqueeze(0)
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
        visual_feat = self._tensor(self.visual_encoder.encode(frame_bgr))
        proprio = self._tensor(self._proprio(sensors, motion, action, dt)).unsqueeze(0)

        surprise, loss = self._train_transition(visual_feat, proprio, action)
        with torch.no_grad():
            visual_z = self.net.visual_embedding(visual_feat)
            z, new_hidden = self.net.encode(visual_feat, proprio, self.hidden)
        self.hidden = new_hidden.detach()

        rnd_norm = self._rnd_update(z.detach())
        visual_mem_dist, visual_cluster_idx = self.visual_memory.query(visual_z.detach())
        path_mem_dist = 0.0
        path_cluster_idx = None
        path_new_cluster = False
        path_mem_norm = 0.0
        if self.step >= PCVM_WARMUP_STEPS:
            visual_cluster_idx, visual_new_cluster = self.visual_memory.update(visual_z.detach(), visual_mem_dist, visual_cluster_idx, self.step)
        else:
            visual_new_cluster = False

        visual_mem_norm = float(np.clip(visual_mem_dist / DINO3_VIS_MEMORY_NORM_DIST, 0.0, 1.0))
        mem_dist = visual_mem_dist
        mem_norm = visual_mem_norm
        new_cluster = bool(visual_new_cluster)
        novelty = float(np.clip(
            PCVM_VIS_MEMORY_WEIGHT * visual_mem_norm
            + PCVM_RND_WEIGHT * rnd_norm
            + PCVM_SURPRISE_WEIGHT * surprise,
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

        self.prev_visual = visual_feat.detach()
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
            'pcvm_bank_size': 0,
            'pcvm_visual_bank_size': len(self.visual_memory.bank),
            'pcvm_loss': loss,
            'pcvm_pose': [self.pose_x, self.pose_y, self.yaw_rad],
            'pcvm_visual': f'dinov3-onnx:{self.visual_encoder.variant}:{self.visual_encoder.input_size}',
            'step': self.step,
        }
