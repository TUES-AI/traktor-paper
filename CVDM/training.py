from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from CVDM.config import CVDMConfig
from CVDM.memory import PhiKNNMemory
from CVDM.models import ControllableVisualDynamics
from CVDM.normalization import RunningMean
from CVDM.observation import candidate_clearance_scores, normalize_candidate_scores
from CVDM.replay import CVDMTransition, TransitionReplayBuffer


class CVDMTrainer:
    def __init__(self, config: CVDMConfig | None = None, device: str | torch.device | None = None):
        self.config = config or CVDMConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ControllableVisualDynamics(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.model.encoder.parameters())
            + list(self.model.forward_model.parameters())
            + list(self.model.inverse_model.parameters()),
            lr=self.config.lr,
        )
        self.rnd_optimizer = torch.optim.Adam(self.model.rnd.predictor.parameters(), lr=self.config.rnd_lr)
        self.memory = PhiKNNMemory(self.config)
        self.rnd_norm = RunningMean()
        self.surprise_norm = RunningMean()
        self.step = 0
        self.last_losses: dict[str, float] = {}

    def _tensor_2d(self, x: np.ndarray | list[float] | torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            t = x.to(self.device, dtype=torch.float32)
        else:
            t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        return t

    @torch.no_grad()
    def encode_np(self, dino: np.ndarray, ranges: np.ndarray, last_action: np.ndarray) -> np.ndarray:
        phi = self.model.encode(self._tensor_2d(dino), self._tensor_2d(ranges), self._tensor_2d(last_action))
        return phi.squeeze(0).detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def policy_observation(
        self,
        dino: np.ndarray,
        ranges: np.ndarray,
        last_action: np.ndarray,
        distances: dict,
        transition_surprise: float = 0.0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        dino_t = self._tensor_2d(dino)
        ranges_t = self._tensor_2d(ranges)
        last_action_t = self._tensor_2d(last_action)
        phi = self.model.encode(dino_t, ranges_t, last_action_t)
        candidates = torch.as_tensor(
            np.asarray(self.config.candidate_actions, dtype=np.float32).reshape(-1, self.config.action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        pred = self.model.predict_candidates(phi, candidates)
        flat = pred.reshape(-1, self.config.phi_dim)
        rnd_raw = self.model.rnd.error(flat).reshape(1, -1)
        density_raw = self.memory.knn_distance_torch(flat).reshape(1, -1) / max(1e-6, self.config.memory_norm_distance)
        candidate_scores_t = normalize_candidate_scores(0.5 * rnd_raw + 0.5 * density_raw).clamp(0.0, 1.0)
        candidate_scores = candidate_scores_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
        clearance = candidate_clearance_scores(distances, self.config)
        rnd_current_raw = float(self.model.rnd.error(phi).detach().cpu().item())
        rnd_current = self.rnd_norm.normalized(rnd_current_raw)
        surprise_current = self.surprise_norm.normalized(float(transition_surprise))
        obs = np.concatenate(
            [
                phi.squeeze(0).detach().cpu().numpy().astype(np.float32),
                np.asarray(ranges, dtype=np.float32).reshape(-1),
                np.asarray(last_action, dtype=np.float32).reshape(-1),
                candidate_scores,
                clearance,
                np.array([rnd_current, surprise_current], dtype=np.float32),
            ]
        ).astype(np.float32)
        expected = self.config.phi_dim + self.config.range_dim + self.config.action_dim + len(self.config.candidate_actions) * 2 + 2
        if obs.shape != (expected,):
            raise ValueError(f"bad CVDM observation shape {obs.shape}, expected {(expected,)}")
        summary = {
            "phi_norm": float(torch.linalg.vector_norm(phi).detach().cpu().item()),
            "candidate_actions": [float(x) for x in self.config.candidate_actions],
            "candidate_scores": [float(x) for x in candidate_scores],
            "candidate_clearance": [float(x) for x in clearance],
            "rnd_current_raw": rnd_current_raw,
            "rnd_current": float(rnd_current),
            "transition_surprise": float(transition_surprise),
            "transition_surprise_norm": float(surprise_current),
            "density_bank_size": len(self.memory),
        }
        return obs, summary

    def compute_losses(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        phi_t = self.model.encode(batch["dino_t"], batch["range_t"], batch["last_action_t"])
        phi_tp1 = self.model.encode(batch["dino_tp1"], batch["range_tp1"], batch["action_executed"])
        pred_tp1 = self.model.forward_model(phi_t, batch["action_executed"])
        action_hat = self.model.inverse_model(phi_t.detach(), phi_tp1.detach())

        forward_loss = F.mse_loss(pred_tp1, phi_tp1.detach())
        inverse_loss = F.mse_loss(action_hat, batch["action_executed"])
        static_mask = (batch["executed_distance_cm"] < self.config.static_distance_cm).float().view(-1, 1)
        if float(static_mask.sum().detach().cpu().item()) > 0.0:
            static_loss = (static_mask * (phi_t - phi_tp1).pow(2)).sum() / (static_mask.sum() * phi_t.shape[-1]).clamp_min(1.0)
        else:
            static_loss = phi_t.new_tensor(0.0)
        rnd_error = self.model.rnd.error(phi_tp1.detach())
        rnd_loss = rnd_error.mean()
        anti_collapse_loss, anti_collapse_metrics = self._anti_collapse_loss(phi_t, phi_tp1)
        total = (
            self.config.forward_weight * forward_loss
            + self.config.inverse_weight * inverse_loss
            + self.config.static_weight * static_loss
            + self.config.rnd_weight * rnd_loss
            + self.config.anti_collapse_weight * anti_collapse_loss
        )
        metrics = {
            "forward_loss": float(forward_loss.detach().cpu().item()),
            "inverse_loss": float(inverse_loss.detach().cpu().item()),
            "static_loss": float(static_loss.detach().cpu().item()),
            "rnd_loss": float(rnd_loss.detach().cpu().item()),
            "anti_collapse_loss": float(anti_collapse_loss.detach().cpu().item()),
            "loss": float(total.detach().cpu().item()),
            **anti_collapse_metrics,
        }
        return total, metrics

    def _anti_collapse_loss(self, phi_t: torch.Tensor, phi_tp1: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        z = torch.cat([phi_t, phi_tp1], dim=0)
        if z.shape[0] < 2:
            zero = z.new_tensor(0.0)
            return zero, {
                "phi_batch_std_mean": 0.0,
                "phi_batch_std_min": 0.0,
                "phi_mean_direction_norm": float(torch.linalg.vector_norm(z.mean(dim=0)).detach().cpu().item()) if z.numel() else 0.0,
            }
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-6)
        std_loss = F.relu(self.config.anti_collapse_min_std - std).mean()
        mean_direction_norm = torch.linalg.vector_norm(z.mean(dim=0), ord=2)
        mean_loss = mean_direction_norm.pow(2) * self.config.anti_collapse_mean_weight
        return std_loss + mean_loss, {
            "phi_batch_std_mean": float(std.mean().detach().cpu().item()),
            "phi_batch_std_min": float(std.min().detach().cpu().item()),
            "phi_mean_direction_norm": float(mean_direction_norm.detach().cpu().item()),
        }

    @torch.no_grad()
    def transition_error(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        phi_t = self.model.encode(batch["dino_t"], batch["range_t"], batch["last_action_t"])
        phi_tp1 = self.model.encode(batch["dino_tp1"], batch["range_tp1"], batch["action_executed"])
        pred = self.model.forward_model(phi_t, batch["action_executed"])
        return (pred - phi_tp1).pow(2).mean(dim=-1)

    def train_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        loss, metrics = self.compute_losses(batch)
        self.optimizer.zero_grad(set_to_none=True)
        self.rnd_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
        self.rnd_optimizer.step()
        self.last_losses = metrics
        return metrics

    def observe_transition(
        self,
        transition: CVDMTransition,
        replay: TransitionReplayBuffer,
        batch_size: int = 32,
        gradient_steps: int = 1,
    ) -> dict[str, Any]:
        single = replay.single(transition, self.device)
        err_before = float(self.transition_error(single).mean().detach().cpu().item())
        replay.add(transition)

        losses: dict[str, float] = {}
        for _ in range(max(1, int(gradient_steps))):
            batch = replay.sample(batch_size, self.device)
            losses = self.train_batch(batch)

        err_after = float(self.transition_error(single).mean().detach().cpu().item())
        lp = max(0.0, err_before - err_after)
        surprise_norm = self.surprise_norm.update(err_after)

        with torch.no_grad():
            phi_tp1 = self.model.encode(
                self._tensor_2d(transition.dino_tp1),
                self._tensor_2d(transition.range_tp1),
                self._tensor_2d(transition.action_executed),
            )
            rnd_raw = float(self.model.rnd.error(phi_tp1).detach().cpu().item())
            rnd_norm = self.rnd_norm.update(rnd_raw)
        mem_details = self.memory.query_details(phi_tp1)
        mem_dist = float(mem_details["distance"])
        mem_idx = mem_details["index"]
        mem_norm = float(mem_details["norm"])
        memory_update_ok = bool(transition.metadata.get("memory_update_ok", True))
        memory_update_reason = str(transition.metadata.get("memory_update_reason", "ok"))
        update_action = "skipped_invalid_visual_memory"
        update_dist = mem_dist
        update_norm = mem_norm
        new_cluster = False
        if memory_update_ok:
            cluster_id, new_cluster, update_dist, update_norm, update_action = self.memory.update(phi_tp1, self.step)
        else:
            cluster_id = -1 if mem_idx is None else int(mem_idx)
            mem_norm = 0.0
        self.step += 1

        return {
            "forward_error_before": err_before,
            "forward_error_after": err_after,
            "learning_progress": float(lp),
            "transition_surprise_norm": float(surprise_norm),
            "novelty_phi": float(mem_norm),
            "novelty_phi_raw": float(mem_details["norm"]),
            "density_distance": float(mem_dist),
            "density_cluster_id_before": None if mem_idx is None else int(mem_idx),
            "density_cluster_id": int(cluster_id),
            "density_new_cluster": bool(new_cluster),
            "density_update_action": update_action,
            "density_update_ok": bool(memory_update_ok and update_action in {"updated_existing", "created_new"}),
            "density_skip_reason": None if memory_update_ok else memory_update_reason,
            "density_second_distance": mem_details["second_distance"],
            "density_assignment_margin": mem_details["assignment_margin"],
            "density_update_distance": float(update_dist),
            "density_update_norm": float(update_norm),
            "density_bank_size": len(self.memory),
            "rnd_raw": float(rnd_raw),
            "rnd_norm": float(rnd_norm),
            **losses,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rnd_optimizer": self.rnd_optimizer.state_dict(),
            "memory": self.memory.state_dict(),
            "rnd_norm": self.rnd_norm.state_dict(),
            "surprise_norm": self.surprise_norm.state_dict(),
            "step": int(self.step),
            "config": self.config.to_dict(),
        }

    def load_state_dict(self, state: dict[str, Any], *, load_optimizers: bool = True) -> None:
        self.model.load_state_dict(state["model"])
        if load_optimizers:
            if state.get("optimizer") is not None:
                self.optimizer.load_state_dict(state["optimizer"])
            if state.get("rnd_optimizer") is not None:
                self.rnd_optimizer.load_state_dict(state["rnd_optimizer"])
        if state.get("memory") is not None:
            self.memory.load_state_dict(state["memory"])
        if state.get("rnd_norm") is not None:
            self.rnd_norm.load_state_dict(state["rnd_norm"])
        if state.get("surprise_norm") is not None:
            self.surprise_norm.load_state_dict(state["surprise_norm"])
        self.step = int(state.get("step", self.step))
