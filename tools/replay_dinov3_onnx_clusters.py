#!/usr/bin/env python3
"""Short offline DINOv3-ONNX visual clustering experiment.

This intentionally does not plug ONNX into PCVM. It replays saved rover frames,
extracts frozen DINOv3 ViT-S/16 pooler embeddings with ONNX Runtime, clusters
only those image embeddings with the same simple memory-bank rule, and writes a
cluster montage PNG plus a JSONL log.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
import onnxruntime as ort
from PIL import Image, ImageDraw


REPO = 'onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX'


def preprocess(frame_bgr, size):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    return np.transpose(x, (2, 0, 1))[None].astype(np.float32)


def l2_normalize(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return x / (np.linalg.norm(x) + 1e-8)


class MemoryBank:
    def __init__(self, known_dist=0.38, update_rate=0.01, warmup=5):
        self.known_dist = float(known_dist)
        self.update_rate = float(update_rate)
        self.warmup = int(warmup)
        self.bank = []
        self.counts = []

    def query(self, z):
        if not self.bank:
            return 1.0, None
        dists = [float(np.linalg.norm(z - c)) for c in self.bank]
        idx = int(np.argmin(dists))
        return dists[idx], idx

    def update(self, z, dist, idx, step):
        if step < self.warmup:
            return idx, False
        if idx is None or dist > self.known_dist:
            self.bank.append(z.copy())
            self.counts.append(1)
            return len(self.bank) - 1, True
        self.bank[idx] = l2_normalize((1.0 - self.update_rate) * self.bank[idx] + self.update_rate * z)
        self.counts[idx] += 1
        return idx, False


def load_session(variant, threads):
    model_file = f'onnx/{variant}.onnx'
    data_file = f'onnx/{variant}.onnx_data'
    hf_hub_download(REPO, data_file)
    model_path = hf_hub_download(REPO, model_file)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(threads)
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])


def frame_paths(frame_dir, limit=0):
    paths = sorted(Path(frame_dir).glob('step_*.jpg'))
    if limit:
        paths = paths[:limit]
    return paths


def make_sheet(records, out_png, title):
    groups = defaultdict(list)
    for r in records:
        if r['cluster'] is None:
            continue
        groups[r['cluster']].append(r)
    thumb_w, thumb_h = 160, 120
    cols = 10
    label_w = 220
    row_h = thumb_h + 42
    width = label_w + cols * (thumb_w + 8) + 20
    height = 48 + len(groups) * row_h
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill='black')
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999']
    y = 40
    for idx, cid in enumerate(sorted(groups)):
        items = groups[cid]
        picks = [items[round(i * (len(items) - 1) / (cols - 1))] for i in range(cols)] if len(items) > cols else items
        color = colors[idx % len(colors)]
        steps = [r['step'] for r in items]
        draw.rectangle((10, y, 20, y + row_h - 8), fill=color)
        draw.text((28, y + 4), f'cluster {cid}', fill='black')
        draw.text((28, y + 24), f'n={len(items)} steps {min(steps)}-{max(steps)}', fill='black')
        for j, r in enumerate(picks):
            x = label_w + j * (thumb_w + 8)
            im = Image.open(r['frame_path']).convert('RGB')
            im.thumbnail((thumb_w, thumb_h))
            tile = Image.new('RGB', (thumb_w, thumb_h), '#eeeeee')
            tile.paste(im, ((thumb_w - im.width) // 2, (thumb_h - im.height) // 2))
            canvas.paste(tile, (x, y))
            draw.rectangle((x, y, x + thumb_w - 1, y + thumb_h - 1), outline=color, width=3)
            draw.text((x, y + thumb_h + 2), f's{r["step"]:03d} d{r["dist"]:.2f}', fill='black')
        y += row_h
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame-dir', default='results/pcvm_m_from_latest_sac_100_20260510_frames')
    parser.add_argument('--out-dir', default='results/dinov3_onnx_vits16_replay_thresh038')
    parser.add_argument('--variant', choices=['model', 'model_quantized', 'model_q4'], default='model_quantized')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--known-dist', type=float, default=0.38)
    parser.add_argument('--update-rate', type=float, default=0.01)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = load_session(args.variant, args.threads)
    input_name = session.get_inputs()[0].name
    bank = MemoryBank(args.known_dist, args.update_rate)
    records = []
    times = []
    for step, fp in enumerate(frame_paths(args.frame_dir, args.limit), 1):
        frame = cv2.imread(str(fp))
        x = preprocess(frame, args.size)
        t0 = time.perf_counter()
        out = session.run(None, {input_name: x})
        times.append(time.perf_counter() - t0)
        z = l2_normalize(out[1][0])
        dist, idx = bank.query(z)
        cid, new = bank.update(z, dist, idx, step)
        rec = {'step': step, 'frame_path': str(fp), 'cluster': int(cid) if cid is not None else None, 'dist': float(dist), 'new': bool(new)}
        records.append(rec)
        print(json.dumps({'step': step, 'cluster': rec['cluster'], 'bank': len(bank.bank), 'dist': round(float(dist), 4)}), flush=True)

    log_path = out_dir / 'trajectory.jsonl'
    with log_path.open('w') as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + '\n')
    png_path = out_dir / 'dinov3_onnx_visual_clusters.png'
    make_sheet(records, png_path, f'DINOv3 ONNX {args.variant} size={args.size} threshold={args.known_dist}')
    mean = float(np.mean(times)) if times else 0.0
    summary = {
        'frames': len(records),
        'clusters': len(bank.bank),
        'counts': {str(i): int(c) for i, c in enumerate(bank.counts)},
        'mean_seconds': mean,
        'fps': (1.0 / mean) if mean > 0 else 0.0,
        'png': str(png_path),
        'log': str(log_path),
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
