"""Live Flask dashboard for rover visual analysis.

This is intentionally read-only for learning: it visualizes camera/depth/path
state but never feeds depth or dashboard outputs back into SAC or PCVM.
"""

from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class DashboardConfig:
    host: str = '0.0.0.0'
    port: int = 8765
    depth_model: str = 'depth-anything/Depth-Anything-V2-Small-hf'
    depth_every: int = 3
    max_points: int = 1000


class OptionalDepthEstimator:
    def __init__(self, model_name, depth_every=3):
        self.model_name = model_name
        self.depth_every = int(depth_every)
        self.ready = False
        self.error = None
        self.count = 0
        self.last_depth = None
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            self.torch = torch
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device).eval()
            self.ready = True
        except Exception as exc:  # dashboard must not break rover training
            self.error = repr(exc)

    def estimate(self, frame_bgr):
        self.count += 1
        if self.last_depth is not None and self.count % max(1, self.depth_every) != 0:
            return self.last_depth
        if not self.ready:
            self.last_depth = self._opencv_fallback(frame_bgr)
            return self.last_depth
        try:
            import cv2
            from PIL import Image

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            inputs = self.processor(images=image, return_tensors='pt').to(self.device)
            with self.torch.no_grad():
                outputs = self.model(**inputs)
            post = self.processor.post_process_depth_estimation(outputs, target_sizes=[(rgb.shape[0], rgb.shape[1])])
            depth = post[0]['predicted_depth'].detach().float().cpu().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            self.last_depth = (depth * 255).astype(np.uint8)
            return self.last_depth
        except Exception as exc:
            self.error = repr(exc)
            self.ready = False
            self.last_depth = self._opencv_fallback(frame_bgr)
            return self.last_depth

    def _opencv_fallback(self, frame_bgr):
        import cv2

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        # Not true depth: useful only as a visual texture proxy when learned depth is unavailable.
        return cv2.normalize(255 - blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


class RoverVisualDashboard:
    def __init__(self, config=None):
        self.config = config or DashboardConfig()
        self.depth = OptionalDepthEstimator(self.config.depth_model, depth_every=self.config.depth_every)
        self.lock = threading.Lock()
        self.frame_bgr = None
        self.depth_u8 = None
        self.info = {}
        self.path = []
        self.started = False

    def start(self):
        if self.started:
            return
        self.started = True
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def update(self, info, frame_bgr=None):
        if frame_bgr is not None:
            depth_u8 = self.depth.estimate(frame_bgr)
        else:
            depth_u8 = None
        backend = (info or {}).get('backend') or {}
        pose = backend.get('pcvm_pose') or [0.0, 0.0, 0.0]
        with self.lock:
            self.info = info or {}
            if frame_bgr is not None:
                self.frame_bgr = frame_bgr.copy()
            if depth_u8 is not None:
                self.depth_u8 = depth_u8.copy()
            if len(pose) >= 2:
                self.path.append((float(pose[0]), float(pose[1]), float(backend.get('pcvm_novelty') or 0.0)))
                self.path = self.path[-self.config.max_points:]

    def _run(self):
        from flask import Flask, Response, jsonify

        app = Flask(__name__)

        @app.route('/')
        def index():
            return '''
            <html><head><title>Rover PCVM-T Visual Dashboard</title></head>
            <body style="font-family: sans-serif; background: #111; color: #eee;">
            <h2>Rover PCVM-T Visual Dashboard</h2>
            <p>Depth/map/histogram are visualization-only and are not policy inputs.</p>
            <img src="/view.jpg" style="max-width: 100%; border: 1px solid #444;" />
            <pre id="state"></pre>
            <script>
            setInterval(async () => {
              const r = await fetch('/state.json');
              document.getElementById('state').textContent = JSON.stringify(await r.json(), null, 2);
              document.querySelector('img').src = '/view.jpg?t=' + Date.now();
            }, 1000);
            </script></body></html>
            '''

        @app.route('/state.json')
        def state():
            with self.lock:
                backend = (self.info or {}).get('backend') or {}
                return jsonify({
                    'step': self.info.get('step'),
                    'reward': self.info.get('reward'),
                    'reward_terms': self.info.get('reward_terms'),
                    'distances': self.info.get('distances'),
                    'backend': backend,
                    'depth_model': self.config.depth_model,
                    'depth_ready': self.depth.ready,
                    'depth_error': self.depth.error,
                })

        @app.route('/view.jpg')
        def view():
            return Response(self._render_jpeg(), mimetype='image/jpeg')

        app.run(host=self.config.host, port=self.config.port, threaded=True, use_reloader=False)

    def _render_jpeg(self):
        import cv2

        with self.lock:
            frame = None if self.frame_bgr is None else self.frame_bgr.copy()
            depth = None if self.depth_u8 is None else self.depth_u8.copy()
            path = list(self.path)
            info = dict(self.info)
        if frame is None:
            canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(canvas, 'waiting for camera frame', (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        else:
            frame_small = cv2.resize(frame, (426, 320), interpolation=cv2.INTER_AREA)
            if depth is None:
                depth_color = np.zeros_like(frame_small)
            else:
                depth_small = cv2.resize(depth, (426, 320), interpolation=cv2.INTER_AREA)
                depth_color = cv2.applyColorMap(depth_small, cv2.COLORMAP_MAGMA)
            map_img = self._draw_map(path, 426, 320)
            hist_img = self._draw_depth_hist(depth, 426, 320)
            top = np.hstack([frame_small, depth_color, map_img])
            bottom = np.hstack([hist_img, self._draw_text(info, 852, 320)])
            canvas = np.vstack([top, bottom])
        ok, buf = cv2.imencode('.jpg', canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return io.BytesIO(buf.tobytes() if ok else b'').getvalue()

    def _draw_map(self, path, w, h):
        import cv2

        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, 'executed path map', (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        if not path:
            return img
        xs = np.array([p[0] for p in path], dtype=np.float32)
        ys = np.array([p[1] for p in path], dtype=np.float32)
        scale = min((w - 60) / max(1e-3, float(xs.max() - xs.min() + 0.5)), (h - 60) / max(1e-3, float(ys.max() - ys.min() + 0.5)))
        cx = w // 2 - int(scale * float((xs.max() + xs.min()) * 0.5))
        cy = h // 2 + int(scale * float((ys.max() + ys.min()) * 0.5))
        pts = []
        for x, y, nov in path:
            px = int(cx + x * scale)
            py = int(cy - y * scale)
            pts.append((px, py))
            color = (0, int(80 + 175 * nov), int(255 * (1.0 - nov)))
            cv2.circle(img, (px, py), 3, color, -1)
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, (120, 120, 120), 1)
        cv2.circle(img, pts[0], 6, (255, 255, 255), 1)
        cv2.circle(img, pts[-1], 6, (0, 255, 255), -1)
        return img

    def _draw_depth_hist(self, depth, w, h):
        import cv2

        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, 'relative depth histogram', (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        if depth is None:
            return img
        vals = depth.reshape(-1)
        hist, _ = np.histogram(vals, bins=32, range=(0, 255))
        hist = hist.astype(np.float32) / max(1.0, float(hist.max()))
        bw = max(1, (w - 40) // len(hist))
        for i, v in enumerate(hist):
            x0 = 20 + i * bw
            y0 = h - 24
            y1 = int(y0 - v * (h - 70))
            cv2.rectangle(img, (x0, y1), (x0 + bw - 1, y0), (180, 120, 255), -1)
        return img

    def _draw_text(self, info, w, h):
        import cv2

        img = np.zeros((h, w, 3), dtype=np.uint8)
        backend = (info or {}).get('backend') or {}
        lines = [
            f"step: {info.get('step')}",
            f"reward: {info.get('reward')}",
            f"novelty: {backend.get('pcvm_novelty')}",
            f"bank: {backend.get('pcvm_bank_size')} new: {backend.get('pcvm_new_cluster')}",
            f"travel_m: {backend.get('pcvm_travel_m')}",
            f"visual: {backend.get('pcvm_visual')}",
        ]
        y = 28
        for line in lines:
            cv2.putText(img, str(line), (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
            y += 28
        return img
