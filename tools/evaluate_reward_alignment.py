#!/usr/bin/env python3
"""Offline reward-alignment audit for RLxF rover runs.

This is not a training script. It scores recorded trajectories with a deliberately
simple coverage-oriented proxy so we can compare "good human coverage" against
"bad looping/corridor" segments before changing online reward weights.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (float(np.linalg.norm(v)) + 1e-6)


def image_descriptor(path):
    im = Image.open(path).convert('RGB').resize((96, 72))
    arr = np.asarray(im).astype(np.float32) / 255.0
    bins = 8
    idx = np.minimum((arr * bins).astype(np.int32), bins - 1)
    hist = np.zeros((bins, bins, bins), dtype=np.float32)
    for r, g, b in idx.reshape(-1, 3):
        hist[r, g, b] += 1.0
    hist = hist.flatten()
    hist /= float(hist.sum()) + 1e-6
    gray = arr.mean(axis=2)
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    grad = np.sqrt(gx * gx + gy * gy)
    edge_cols = grad.reshape(72, 12, 8).mean(axis=(0, 2))
    small = np.asarray(Image.fromarray((gray * 255).astype(np.uint8)).resize((16, 12))).astype(np.float32).flatten() / 255.0
    return _norm(np.concatenate([hist * 4.0, edge_cols * 2.0, small * 0.5]))


class Memory:
    def __init__(self, threshold=0.55, update=0.01):
        self.threshold = float(threshold)
        self.update = float(update)
        self.centroids = []
        self.last_seen = []

    def query(self, z):
        if not self.centroids:
            return 1.0, None
        d = np.array([np.linalg.norm(z - c) for c in self.centroids], dtype=np.float32)
        i = int(d.argmin())
        return float(d[i]), i

    def update_with(self, z, step):
        d, i = self.query(z)
        new = i is None or d > self.threshold
        if new:
            self.centroids.append(z.copy())
            self.last_seen.append(step)
            return len(self.centroids) - 1, True, d, None
        age = step - self.last_seen[i]
        self.last_seen[i] = step
        self.centroids[i] = _norm((1.0 - self.update) * self.centroids[i] + self.update * z)
        return i, False, d, age


def near_obstacle(row):
    us = row.get('ultrasonic_cm') or row.get('sensor_status', {}).get('ultrasonic_cm') or {}
    vals = []
    for k, lim, w in [('front', 45.0, 1.0), ('left', 25.0, 0.5), ('right', 25.0, 0.5)]:
        v = us.get(k)
        if v is not None:
            vals.append(w * max(0.0, 1.0 - float(v) / lim))
    return max(vals) if vals else 0.0


def command_motion(row):
    cmd = row.get('command') or row.get('sensor_status', {}).get('command') or {}
    label = cmd.get('label') or 'stop'
    left = float(cmd.get('left_speed') or 0.0)
    right = float(cmd.get('right_speed') or 0.0)
    forward = label in ('forward', 'forward_arc_left', 'forward_arc_right')
    backward = label.startswith('backward')
    spin = 'spin' in label or (left > 0 and right > 0 and (cmd.get('left_direction') != cmd.get('right_direction')))
    moving = forward or backward or spin
    return label, moving, forward, spin


def score_wasd(run_dir, stride=1):
    root = Path(run_dir)
    rows = []
    for line in (root / 'telemetry.jsonl').read_text(errors='ignore').splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get('type') == 'frame' and row.get('ok') and row.get('path'):
            rows.append(row)
    mem = Memory()
    reward = 0.0
    terms = {k: 0.0 for k in ('new', 'novelty', 'forward', 'spin_penalty', 'revisit', 'near')}
    events = []
    last_t = None
    for step, row in enumerate(rows[::stride], start=1):
        z = image_descriptor(root / row['path'])
        cid, new, dist, age = mem.update_with(z, step)
        novelty = min(1.0, dist / mem.threshold)
        label, moving, forward, spin = command_motion(row)
        dt = 0.08 if last_t is None else max(0.02, min(0.25, float(row.get('t_wall_s', 0.0)) - last_t))
        last_t = float(row.get('t_wall_s', 0.0))
        near = near_obstacle(row)
        local = 0.0
        if moving:
            local += 0.35 * novelty
            terms['novelty'] += 0.35 * novelty
        if new and moving:
            local += 1.2
            terms['new'] += 1.2
            events.append((step, cid, round(dist, 3), label))
        if forward:
            local += 0.18
            terms['forward'] += 0.18
        if spin and not forward:
            local -= 0.12
            terms['spin_penalty'] += 0.12
        if (not new) and age is not None and age < 120 and moving:
            p = 0.22 * max(0.0, 1.0 - age / 120.0)
            local -= p
            terms['revisit'] += p
        local -= 0.25 * near
        terms['near'] += 0.25 * near
        reward += local
    return {
        'kind': 'wasd',
        'frames': len(rows),
        'scored_steps': math.ceil(len(rows) / stride),
        'reward': reward,
        'reward_per_100': reward / max(1, math.ceil(len(rows) / stride)) * 100.0,
        'clusters': len(mem.centroids),
        'cluster_events': events[:40],
        'terms': terms,
    }


def score_theta_log(path, start=100, end=130):
    all_rows = []
    for line in Path(path).read_text(errors='ignore').splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if 'reward' in row and 'backend' in row:
            all_rows.append(row)
    rows = all_rows[start:end]
    reward = 0.0
    terms = {k: 0.0 for k in ('new', 'novelty', 'distance', 'safe', 'recovery', 'revisit', 'near', 'zero_forward')}
    prev_xy = []
    for row in rows:
        b = row.get('backend') or {}
        rt = row.get('reward_terms') or {}
        dist_cm = float(rt.get('executed_distance_cm') or 0.0)
        novelty = float(rt.get('path_novelty_raw') or rt.get('novelty_raw') or 0.0)
        new = bool(rt.get('path_new_cluster') or rt.get('visual_new_cluster') or b.get('pcvm_new_cluster'))
        local = 0.0
        if dist_cm > 3.0:
            local += 0.35 * min(1.0, novelty)
            local += 0.70 * min(1.0, dist_cm / 120.0)
            local += 0.25
            terms['novelty'] += 0.35 * min(1.0, novelty)
            terms['distance'] += 0.70 * min(1.0, dist_cm / 120.0)
            terms['safe'] += 0.25
        else:
            local -= 0.25
            terms['zero_forward'] += 0.25
        if new and dist_cm > 3.0:
            local += 1.2
            terms['new'] += 1.2
        if rt.get('recovery_reverse') or rt.get('recovery_turn') or row.get('recovery'):
            local -= 0.35
            terms['recovery'] += 0.35
        near = float(rt.get('near_obstacle_raw') or 0.0)
        local -= 0.20 * near
        terms['near'] += 0.20 * near
        pose = b.get('pcvm_pose') or [None, None]
        if pose[0] is not None:
            x, y = float(pose[0]), float(pose[1])
            if prev_xy:
                md = min(math.hypot(x - px, y - py) for px, py in prev_xy[-25:])
                if md < 0.45:
                    p = 0.45 * (1.0 - md / 0.45)
                    local -= p
                    terms['revisit'] += p
            prev_xy.append((x, y))
        reward += local
    return {
        'kind': 'theta_log',
        'rows_total': len(all_rows),
        'slice': [start, end],
        'steps': len(rows),
        'reward': reward,
        'reward_per_100': reward / max(1, len(rows)) * 100.0,
        'terms': terms,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wasd-run', default='data/manual_runs/wasd_20260506_210844_384674us')
    ap.add_argument('--theta-log', default='results/pcvm_theta_front_rlxf_train_rerun.jsonl')
    ap.add_argument('--theta-start', type=int, default=100)
    ap.add_argument('--theta-end', type=int, default=130)
    ap.add_argument('--stride', type=int, default=1)
    args = ap.parse_args()
    print(json.dumps(score_wasd(args.wasd_run, args.stride), indent=2, sort_keys=True))
    print(json.dumps(score_theta_log(args.theta_log, args.theta_start, args.theta_end), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
