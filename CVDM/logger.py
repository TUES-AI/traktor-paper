from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    return value


class CVDMRunLogger:
    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.frames_dir = self.out_dir / "frames"
        self.models_dir = self.out_dir / "models"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.out_dir / "transitions.jsonl"
        self.json_path = self.out_dir / "transitions.json"
        self.manifest_path = self.out_dir / "manifest.json"
        self.records: list[dict[str, Any]] = []
        self._jsonl = self.jsonl_path.open("a", buffering=1)
        self.frame_count = 0

    def relative(self, path: str | Path) -> str:
        path = Path(path)
        try:
            return str(path.relative_to(self.out_dir))
        except ValueError:
            return str(path)

    def save_frame(self, frame_bgr, label: str) -> dict[str, str | int]:
        import cv2

        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in label)
        path = self.frames_dir / f"frame_{self.frame_count:06d}_{safe}.jpg"
        self.frame_count += 1
        cv2.imwrite(str(path), frame_bgr)
        return {
            "path": str(path),
            "relative_path": self.relative(path),
            "frame_index": self.frame_count - 1,
        }

    def append(self, record: dict[str, Any]) -> None:
        clean = to_jsonable(record)
        self.records.append(clean)
        self._jsonl.write(json.dumps(clean, sort_keys=True) + "\n")

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self.manifest_path.write_text(json.dumps(to_jsonable(manifest), indent=2, sort_keys=True))

    def finalize(self) -> None:
        self.json_path.write_text(json.dumps(to_jsonable(self.records), indent=2, sort_keys=True))
        self._jsonl.flush()
        self._jsonl.close()
