from pathlib import Path

p = Path('embedded/scripts/train_real_predictive_sac.py')
s = p.read_text()

if 'class DinoRNDReward' not in s:
    marker = ")\n\n\nclass RealPredictiveSACEnv"
    insert = """ )


class DinoRNDReward:
    \"\"\"Reward-only DINOv3 RND. DINO never enters the policy observation.\"\"\"
    def __init__(self, weight=0.0, variant='model_q4', size=224, threads=2, lr=5e-5):
        self.weight = float(weight)
        self.variant = variant
        self.size = int(size)
        self.threads = int(threads)
        self.lr = float(lr)
        self.enabled = self.weight > 0.0
        self.session = None
        self.input_name = None
        self.torch = None
        self.target = None
        self.pred = None
        self.opt = None
        self.n = 0
        self.mean = 0.0
        if self.enabled:
            self._load_session()

    def _load_session(self):
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort
        repo = 'onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX'
        model_file = f'onnx/{self.variant}.onnx'
        data_file = f'onnx/{self.variant}.onnx_data'
        try:
            hf_hub_download(repo, data_file)
        except Exception:
            pass
        model_path = hf_hub_download(repo, model_file)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, frame):
        import cv2
        import numpy as np
        if frame is None or getattr(frame, 'ndim', 0) != 3 or frame.shape[2] < 3:
            return None
        rgb = frame[:, :, :3]
        rgb = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        return np.transpose(x, (2, 0, 1))[None].astype(np.float32)

    def _init_rnd(self, dim):
        import torch
        import torch.nn as nn
        self.torch = torch
        self.target = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 128))
        self.pred = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128))
        for param in self.target.parameters():
            param.requires_grad = False
        self.opt = torch.optim.Adam(self.pred.parameters(), lr=self.lr)

    def observe_reward(self, frame):
        if not self.enabled:
            return 0.0, {'dino_rnd_enabled': False}
        try:
            import numpy as np
            import torch.nn.functional as F
            x = self._preprocess(frame)
            if x is None:
                return 0.0, {'dino_rnd_enabled': True, 'dino_rnd_error': 'no_frame'}
            out = self.session.run(None, {self.input_name: x})
            z = np.asarray(out[1][0], dtype=np.float32).reshape(-1)
            z = z / (np.linalg.norm(z) + 1e-8)
            if self.target is None:
                self._init_rnd(int(z.shape[0]))
            zt = self.torch.as_tensor(z, dtype=self.torch.float32).unsqueeze(0)
            with self.torch.no_grad():
                target = self.target(zt)
            pred = self.pred(zt)
            loss = F.mse_loss(pred, target.detach())
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            raw = float(loss.detach().item())
            self.n += 1
            self.mean += (raw - self.mean) / self.n
            norm = float(np.clip(raw / (self.mean + 1e-8), 0.0, 3.0) / 3.0)
            reward = self.weight * norm
            return reward, {
                'dino_rnd_enabled': True,
                'dino_rnd_raw': raw,
                'dino_rnd_mean': self.mean,
                'dino_rnd_norm': norm,
                'dino_rnd_reward': reward,
                'dino_rnd_variant': self.variant,
            }
        except Exception as exc:
            return 0.0, {'dino_rnd_enabled': True, 'dino_rnd_error': repr(exc)}


class RealPredictiveSACEnv"""
    s = s.replace(marker, insert, 1)

if 'self.dino_rnd = DinoRNDReward' not in s:
    s = s.replace(
        '        self.last_reward_terms = {}\n',
        "        self.last_reward_terms = {}\n        self.dino_rnd = DinoRNDReward(\n            weight=args.dino_rnd_weight,\n            variant=args.dino_rnd_variant,\n            size=args.dino_rnd_size,\n            threads=args.dino_rnd_threads,\n            lr=args.dino_rnd_lr,\n        )\n",
        1,
    )

if 'dino_reward, dino_terms = self.dino_rnd.observe_reward' not in s:
    s = s.replace(
        """        reward = self._reward(execution, backend, recovery)
        self._update_path_memory(backend)
        info = {
""",
        """        reward = self._reward(execution, backend, recovery)
        dino_reward, dino_terms = self.dino_rnd.observe_reward(getattr(self.obs_builder, 'last_frame', None))
        reward += dino_reward
        if dino_terms:
            self.last_reward_terms.update(dino_terms)
        self._update_path_memory(backend)
        info = {
""",
        1,
    )

if '--dino-rnd-weight' not in s:
    marker = "    parser.add_argument('--viz-depth-model', default='depth-anything/Depth-Anything-V2-Small-hf')\n"
    add = """    parser.add_argument('--dino-rnd-weight', type=float, default=0.0, help='Reward-only DINO RND bonus weight; DINO is not added to policy observation')
    parser.add_argument('--dino-rnd-variant', choices=['model', 'model_quantized', 'model_q4'], default='model_q4')
    parser.add_argument('--dino-rnd-size', type=int, default=224)
    parser.add_argument('--dino-rnd-threads', type=int, default=2)
    parser.add_argument('--dino-rnd-lr', type=float, default=5e-5)
"""
    s = s.replace(marker, marker + add, 1)

p.write_text(s)
