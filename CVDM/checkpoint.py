from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from CVDM.replay import TransitionReplayBuffer
from CVDM.training import CVDMTrainer


def save_cvdm_checkpoint(
    trainer: CVDMTrainer,
    path: str | Path,
    *,
    replay_state: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format": "cvdm_state_v1",
        **trainer.state_dict(),
        "replay": replay_state,
        "extra": extra or {},
    }
    torch.save(state, path)
    return str(path)


def save_individual_model_params(trainer: CVDMTrainer, model_dir: str | Path) -> dict[str, str]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    modules = {
        "cvdm_encoder.pt": trainer.model.encoder.state_dict(),
        "cvdm_forward_dynamics.pt": trainer.model.forward_model.state_dict(),
        "cvdm_inverse_dynamics.pt": trainer.model.inverse_model.state_dict(),
        "cvdm_rnd_target.pt": trainer.model.rnd.target.state_dict(),
        "cvdm_rnd_predictor.pt": trainer.model.rnd.predictor.state_dict(),
    }
    saved = {}
    for name, state in modules.items():
        path = model_dir / name
        torch.save(state, path)
        saved[name] = str(path)
    torch.save(
        {
            "dynamics_optimizer": trainer.optimizer.state_dict(),
            "rnd_optimizer": trainer.rnd_optimizer.state_dict(),
            "density_memory": trainer.memory.state_dict(),
            "rnd_norm": trainer.rnd_norm.state_dict(),
            "surprise_norm": trainer.surprise_norm.state_dict(),
        },
        model_dir / "cvdm_training_state.pt",
    )
    saved["cvdm_training_state.pt"] = str(model_dir / "cvdm_training_state.pt")
    (model_dir / "cvdm_config.json").write_text(json.dumps(trainer.config.to_dict(), indent=2, sort_keys=True))
    saved["cvdm_config.json"] = str(model_dir / "cvdm_config.json")
    return saved


def _torch_load(path: str | Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_cvdm_checkpoint(
    trainer: CVDMTrainer,
    path: str | Path,
    *,
    replay: TransitionReplayBuffer | None = None,
    map_location="cpu",
    load_optimizers: bool = True,
) -> dict[str, Any]:
    state = _torch_load(path, map_location=map_location)
    trainer.load_state_dict(state, load_optimizers=load_optimizers)
    if replay is not None and state.get("replay") is not None:
        replay.load_state_dict(state["replay"])
    return state
