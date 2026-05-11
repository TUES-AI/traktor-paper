from __future__ import annotations

import numpy as np

from CVDM.normalization import normalize_l2_np


DINO3_ONNX_REPO = "onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX"
DINO3_ONNX_VARIANT = "model_quantized"
DINO3_INPUT_SIZE = 336
DINO3_FEATURE_DIM = 384


class FrozenDINOv3ONNXEncoder:
    """Frozen quantized DINOv3 ONNX encoder used by the rover experiments."""

    def __init__(
        self,
        repo: str = DINO3_ONNX_REPO,
        variant: str = DINO3_ONNX_VARIANT,
        input_size: int = DINO3_INPUT_SIZE,
        threads: int = 4,
    ) -> None:
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort

        self.repo = repo
        self.variant = variant
        self.input_size = int(input_size)
        self.feature_dim = DINO3_FEATURE_DIM
        hf_hub_download(repo, f"onnx/{variant}.onnx_data")
        model_path = hf_hub_download(repo, f"onnx/{variant}.onnx")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(threads)
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        return np.transpose(x, (2, 0, 1))[None].astype(np.float32)

    def encode(self, frame_bgr: np.ndarray) -> np.ndarray:
        x = self.preprocess_frame(frame_bgr)
        outputs = self.session.run(None, {self.input_name: x})
        feat = None
        for out in outputs:
            arr = np.asarray(out, dtype=np.float32)
            if arr.shape[-1] == self.feature_dim:
                feat = arr.reshape(-1, self.feature_dim)[0]
                break
        if feat is None:
            feat = np.asarray(outputs[-1], dtype=np.float32).reshape(-1)[-self.feature_dim :]
        return normalize_l2_np(feat)

    def metadata(self) -> dict[str, object]:
        return {
            "kind": "dinov3_onnx",
            "repo": self.repo,
            "variant": self.variant,
            "input_size": self.input_size,
            "feature_dim": self.feature_dim,
        }


class HashVisualEncoder:
    """Fast deterministic visual encoder for local smoke tests without ONNX."""

    def __init__(self, feature_dim: int = DINO3_FEATURE_DIM, seed: int = 1234) -> None:
        self.feature_dim = int(feature_dim)
        rng = np.random.default_rng(seed)
        self._proj = rng.standard_normal((24 * 16 * 3, self.feature_dim), dtype=np.float32) / np.sqrt(24 * 16 * 3)

    def encode(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2

        small = cv2.resize(frame_bgr, (24, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        flat = small.reshape(-1)
        return normalize_l2_np(flat @ self._proj)

    def metadata(self) -> dict[str, object]:
        return {"kind": "hash", "feature_dim": self.feature_dim}


def make_visual_encoder(kind: str = "dino3", input_size: int = DINO3_INPUT_SIZE, threads: int = 4):
    kind = str(kind).lower()
    if kind in {"dino", "dino3", "dinov3", "onnx"}:
        return FrozenDINOv3ONNXEncoder(input_size=input_size, threads=threads)
    if kind in {"hash", "dummy", "smoke"}:
        return HashVisualEncoder()
    raise ValueError(f"unknown visual encoder kind: {kind}")
