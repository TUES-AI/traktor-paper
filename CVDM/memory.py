from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from CVDM.config import CVDMConfig


class PhiKNNMemory:
    """Bounded prototype memory over normalized controllable latents."""

    def __init__(self, config: CVDMConfig):
        self.maxlen = int(config.memory_size)
        self.known_distance = float(config.memory_known_distance)
        self.norm_distance = float(config.memory_norm_distance)
        self.update_rate = float(config.memory_update_rate)
        self.min_assignment_margin = float(config.memory_min_assignment_margin)
        self.bank: list[torch.Tensor] = []
        self.counts: list[int] = []
        self.last_seen: list[int] = []

    def __len__(self) -> int:
        return len(self.bank)

    def _stack(self, device: torch.device) -> torch.Tensor:
        return torch.stack([x.to(device) for x in self.bank], dim=0)

    def query(self, phi: torch.Tensor) -> tuple[float, int | None, float]:
        details = self.query_details(phi)
        return details["distance"], details["index"], details["norm"]

    def query_details(self, phi: torch.Tensor) -> dict[str, float | int | None]:
        z = F.normalize(phi.detach().reshape(-1), dim=0).cpu()
        if not self.bank:
            return {
                "distance": self.norm_distance,
                "index": None,
                "norm": 1.0,
                "second_distance": None,
                "assignment_margin": None,
            }
        bank = torch.stack(self.bank, dim=0)
        dists = torch.linalg.vector_norm(bank - z.unsqueeze(0), dim=1)
        idx = int(dists.argmin().item())
        dist = float(dists[idx].item())
        second = None
        margin = None
        if dists.numel() > 1:
            vals = torch.topk(dists, k=2, largest=False).values
            second = float(vals[1].item())
            margin = float(second - dist)
        return {
            "distance": dist,
            "index": idx,
            "norm": float(np.clip(dist / max(1e-8, self.norm_distance), 0.0, 1.0)),
            "second_distance": second,
            "assignment_margin": margin,
        }

    def knn_distance_torch(self, phi: torch.Tensor) -> torch.Tensor:
        if not self.bank:
            return torch.full((phi.shape[0],), self.norm_distance, dtype=phi.dtype, device=phi.device)
        bank = self._stack(phi.device).to(dtype=phi.dtype)
        phi_n = F.normalize(phi, dim=-1)
        dists = torch.linalg.vector_norm(phi_n[:, None, :] - bank[None, :, :], dim=-1)
        return dists.min(dim=1).values

    def update(self, phi: torch.Tensor, step: int) -> tuple[int, bool, float, float, str]:
        z = F.normalize(phi.detach().reshape(-1), dim=0).cpu()
        details = self.query_details(z)
        dist = float(details["distance"])
        idx = details["index"]
        norm = float(details["norm"])
        margin = details["assignment_margin"]
        if idx is not None and dist < self.known_distance:
            if margin is not None and float(margin) < self.min_assignment_margin:
                return int(idx), False, dist, norm, "skipped_ambiguous_assignment"
            eta = min(self.update_rate, 1.0 / (self.counts[idx] + 1))
            self.bank[idx] = F.normalize((1.0 - eta) * self.bank[idx] + eta * z, dim=0).detach().cpu()
            self.counts[idx] += 1
            self.last_seen[idx] = int(step)
            return idx, False, dist, norm, "updated_existing"
        if len(self.bank) >= self.maxlen:
            evict = int(np.argmin(self.last_seen))
            self.bank.pop(evict)
            self.counts.pop(evict)
            self.last_seen.pop(evict)
        self.bank.append(z.detach().cpu())
        self.counts.append(1)
        self.last_seen.append(int(step))
        return len(self.bank) - 1, True, dist, norm, "created_new"

    def state_dict(self) -> dict[str, object]:
        return {
            "bank": [x.detach().cpu() for x in self.bank],
            "counts": list(self.counts),
            "last_seen": list(self.last_seen),
            "maxlen": self.maxlen,
            "known_distance": self.known_distance,
            "norm_distance": self.norm_distance,
            "update_rate": self.update_rate,
            "min_assignment_margin": self.min_assignment_margin,
        }

    def load_state_dict(self, state: dict) -> None:
        self.bank = [x.detach().cpu() for x in state.get("bank", [])]
        self.counts = [int(x) for x in state.get("counts", [])]
        self.last_seen = [int(x) for x in state.get("last_seen", [])]
        self.maxlen = int(state.get("maxlen", self.maxlen))
        self.known_distance = float(state.get("known_distance", self.known_distance))
        self.norm_distance = float(state.get("norm_distance", self.norm_distance))
        self.update_rate = float(state.get("update_rate", self.update_rate))
        self.min_assignment_margin = float(state.get("min_assignment_margin", self.min_assignment_margin))
