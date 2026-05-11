from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class CVDMTransition:
    dino_t: np.ndarray
    range_t: np.ndarray
    last_action_t: np.ndarray
    action_executed: np.ndarray
    dino_tp1: np.ndarray
    range_tp1: np.ndarray
    executed_distance_cm: float
    front_after_cm: float
    contact_or_stall: bool
    recovery: bool
    metadata: dict[str, Any]


class TransitionReplayBuffer:
    def __init__(self, maxlen: int = 5000):
        self.maxlen = int(maxlen)
        self.items: list[CVDMTransition] = []
        self.next_idx = 0

    def __len__(self) -> int:
        return len(self.items)

    def add(self, transition: CVDMTransition) -> None:
        if len(self.items) < self.maxlen:
            self.items.append(transition)
        else:
            self.items[self.next_idx] = transition
            self.next_idx = (self.next_idx + 1) % self.maxlen

    def _batch_from_items(self, items: list[CVDMTransition], device: torch.device) -> dict[str, torch.Tensor]:
        def stack(name: str, dtype=np.float32) -> torch.Tensor:
            return torch.as_tensor(np.stack([np.asarray(getattr(x, name), dtype=dtype) for x in items]), dtype=torch.float32, device=device)

        return {
            "dino_t": stack("dino_t"),
            "range_t": stack("range_t"),
            "last_action_t": stack("last_action_t"),
            "action_executed": stack("action_executed"),
            "dino_tp1": stack("dino_tp1"),
            "range_tp1": stack("range_tp1"),
            "executed_distance_cm": torch.as_tensor([x.executed_distance_cm for x in items], dtype=torch.float32, device=device),
            "front_after_cm": torch.as_tensor([x.front_after_cm for x in items], dtype=torch.float32, device=device),
            "contact_or_stall": torch.as_tensor([float(x.contact_or_stall) for x in items], dtype=torch.float32, device=device),
            "recovery": torch.as_tensor([float(x.recovery) for x in items], dtype=torch.float32, device=device),
        }

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        if not self.items:
            raise RuntimeError("cannot sample empty CVDM replay")
        n = min(int(batch_size), len(self.items))
        idx = np.random.choice(len(self.items), size=n, replace=False)
        return self._batch_from_items([self.items[int(i)] for i in idx], device)

    def single(self, transition: CVDMTransition, device: torch.device) -> dict[str, torch.Tensor]:
        return self._batch_from_items([transition], device)

    def state_dict(self) -> dict[str, object]:
        return {
            "maxlen": self.maxlen,
            "next_idx": self.next_idx,
            "size": len(self.items),
            "items": self.items,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.maxlen = int(state.get("maxlen", self.maxlen))
        self.next_idx = int(state.get("next_idx", 0))
        self.items = list(state.get("items", []))
        if len(self.items) > self.maxlen:
            self.items = self.items[-self.maxlen :]
        if self.items:
            self.next_idx %= len(self.items)
        else:
            self.next_idx = 0
