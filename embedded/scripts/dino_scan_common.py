#!/usr/bin/env python3
"""Shared helpers for active DINOv3 scan experiments."""

import json
from pathlib import Path
import time

import numpy as np

import _paths  # noqa: F401
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from VMM.pcvm import clamp
from VMM.pcvm_d3 import DINO3_VIS_KNOWN_DIST, DINO3_VIS_MEMORY_NORM_DIST, DINO3L2MemoryBank, DINOv3ONNXVisualEncoder


ULTRA_MAX_CM = 400.0


def norm_distance(cm):
    if cm is None:
        return 1.0
    return clamp(float(cm) / ULTRA_MAX_CM, 0.0, 1.0)


def normalize_np(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return x / (float(np.linalg.norm(x)) + 1e-8)


class NumpyDINOVisualMemory:
    def __init__(self, known_dist=DINO3_VIS_KNOWN_DIST, norm_dist=DINO3_VIS_MEMORY_NORM_DIST, update_rate=0.01, warmup=5):
        self.known_dist = float(known_dist)
        self.norm_dist = float(norm_dist)
        self.update_rate = float(update_rate)
        self.warmup = int(warmup)
        self.bank = []
        self.counts = []
        self.last_seen = []

    def query(self, z):
        z = normalize_np(z)
        if not self.bank:
            return 1.0, None, 1.0 / self.norm_dist
        dists = [float(np.linalg.norm(z - c)) for c in self.bank]
        idx = int(np.argmin(dists))
        dist = dists[idx]
        return dist, idx, float(np.clip(dist / self.norm_dist, 0.0, 1.0))

    def update(self, z, step):
        z = normalize_np(z)
        dist, idx, norm = self.query(z)
        if step < self.warmup:
            return idx, False, dist, norm
        if idx is None or dist > self.known_dist:
            self.bank.append(z.copy())
            self.counts.append(1)
            self.last_seen.append(int(step))
            return len(self.bank) - 1, True, dist, norm
        eta = min(self.update_rate, 1.0 / (self.counts[idx] + 1))
        self.bank[idx] = normalize_np((1.0 - eta) * self.bank[idx] + eta * z)
        self.counts[idx] += 1
        self.last_seen[idx] = int(step)
        return idx, False, dist, norm


class DINOScanner:
    def __init__(self, input_size=336, known_dist=DINO3_VIS_KNOWN_DIST, norm_dist=DINO3_VIS_MEMORY_NORM_DIST, warmup=0):
        self.encoder = DINOv3ONNXVisualEncoder(input_size=input_size)
        self.memory = NumpyDINOVisualMemory(known_dist=known_dist, norm_dist=norm_dist, warmup=warmup)

    def encode(self, frame_bgr):
        return normalize_np(self.encoder.encode(frame_bgr)[0])

    def query_frame(self, frame_bgr):
        z = self.encode(frame_bgr)
        dist, cid, norm = self.memory.query(z)
        return {'embedding': z, 'dist': dist, 'cluster_id': cid, 'novelty': norm, 'bank_size': len(self.memory.bank)}

    def update_frame(self, frame_bgr, step):
        z = self.encode(frame_bgr)
        cid, new, dist, norm = self.memory.update(z, step)
        return {'embedding': z, 'dist': dist, 'cluster_id': cid, 'new_cluster': new, 'novelty': norm, 'bank_size': len(self.memory.bank)}


def read_distances(safety):
    return safety.read_distances()


def distance_features(distances):
    return np.array([
        norm_distance(distances.get('left')),
        norm_distance(distances.get('right')),
        norm_distance(distances.get('front')),
    ], dtype=np.float32)


def save_frame(frame, path):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)


def make_visual_cluster_sheet(log_path, frame_dir, out_png):
    from collections import defaultdict
    from PIL import Image, ImageDraw

    rows = []
    for line in Path(log_path).read_text(errors='ignore').splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    groups = defaultdict(list)
    for row in rows:
        cid = (row.get('post_update') or row.get('chosen') or {}).get('cluster_id')
        if cid is not None:
            groups[int(cid)].append(row)
    if not groups:
        return None
    thumb_w, thumb_h = 160, 120
    cols = 10
    label_w = 220
    row_h = thumb_h + 42
    width = label_w + cols * (thumb_w + 8) + 20
    height = 48 + len(groups) * row_h
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f'{Path(log_path).stem} visual clusters', fill='black')
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999']
    y = 40
    for idx, cid in enumerate(sorted(groups)):
        items = groups[cid]
        picks = [items[round(i * (len(items) - 1) / (cols - 1))] for i in range(cols)] if len(items) > cols else items
        color = colors[idx % len(colors)]
        steps = [int(r.get('step') or 0) for r in items]
        draw.rectangle((10, y, 20, y + row_h - 8), fill=color)
        draw.text((28, y + 4), f'cluster {cid}', fill='black')
        draw.text((28, y + 24), f'n={len(items)} steps {min(steps)}-{max(steps)}', fill='black')
        for j, row in enumerate(picks):
            step = int(row.get('step') or 0)
            fp = frame_dir / f'step_{step:04d}_post.jpg'
            if not fp.exists():
                fp = frame_dir / f'step_{step:04d}_chosen.jpg'
            x = label_w + j * (thumb_w + 8)
            try:
                im = Image.open(fp).convert('RGB')
                im.thumbnail((thumb_w, thumb_h))
                tile = Image.new('RGB', (thumb_w, thumb_h), '#eeeeee')
                tile.paste(im, ((thumb_w - im.width) // 2, (thumb_h - im.height) // 2))
            except Exception:
                tile = Image.new('RGB', (thumb_w, thumb_h), '#cccccc')
            canvas.paste(tile, (x, y))
            draw.rectangle((x, y, x + thumb_w - 1, y + thumb_h - 1), outline=color, width=3)
            score = float(((row.get('post_update') or row.get('chosen') or {}).get('novelty') or 0.0))
            draw.text((x, y + thumb_h + 2), f's{step:03d} nov{score:.2f}', fill='black')
        y += row_h
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    return str(out_png)


def summarize_log(log_path):
    rows = []
    for line in Path(log_path).read_text(errors='ignore').splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    if not rows:
        return {}
    rewards = [float(r.get('reward') or 0.0) for r in rows]
    clusters = sorted({((r.get('post_update') or r.get('chosen') or {}).get('cluster_id')) for r in rows if ((r.get('post_update') or r.get('chosen') or {}).get('cluster_id')) is not None})
    execs = [r.get('execution') for r in rows if r.get('execution')]
    return {
        'steps': len(rows),
        'reward_sum': float(sum(rewards)),
        'positive_steps': int(sum(1 for r in rewards if r > 0)),
        'clusters': len(clusters),
        'executions': len(execs),
        'timestamp': time.time(),
    }
