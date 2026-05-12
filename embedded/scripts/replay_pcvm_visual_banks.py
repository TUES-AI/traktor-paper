#!/usr/bin/env python3
"""Replay saved rover frames through PCVM/PCVM-M/PCVM-D and plot memory clusters.

Use this when a real run already covered useful rooms/corridors and we only want
to test whether a visual backend creates meaningful memory-bank clusters.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

import _paths  # noqa: F401
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from VMM.pcvm import clamp
from wasd_pcvm_visual_banks import make_cluster_sheets, make_model


ULTRA_MAX_CM = 400.0


def norm_distance(cm):
    if cm is None:
        return 1.0
    return clamp(float(cm) / ULTRA_MAX_CM, 0.0, 1.0)


def load_rows(path):
    rows = []
    for line in Path(path).read_text(errors='ignore').splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def find_frame(row, frame_dir):
    step = int(row.get('step') or 0)
    explicit = row.get('frame_path')
    candidates = []
    if explicit:
        p = Path(explicit)
        candidates.append(p)
        candidates.append(frame_dir / p.name)
    candidates.append(frame_dir / f'step_{step:04d}.jpg')
    for path in candidates:
        if path.exists():
            return path
    return None


def row_action(row, action_dim):
    vals = row.get('executed_action_for_pcvm') or row.get('action') or []
    arr = np.asarray(vals, dtype=np.float32).reshape(-1)[:action_dim]
    if len(arr) < action_dim:
        arr = np.pad(arr, (0, action_dim - len(arr))).astype(np.float32)
    return arr


def row_sensors(row):
    distances = row.get('distances') or {}
    return np.array([
        norm_distance(distances.get('left')),
        norm_distance(distances.get('right')),
        norm_distance(distances.get('front')),
    ], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser(description='Replay saved rover frames through a PCVM visual backend.')
    p.add_argument('--backend', choices=['pcvm-d', 'pcvm-j', 'pcvm-m', 'pcvm'], default='pcvm-d')
    p.add_argument('--log', default='results/pcvm_m_from_latest_sac_100_20260510.jsonl')
    p.add_argument('--frame-dir', default='results/pcvm_m_from_latest_sac_100_20260510_frames')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--action-dim', type=int, default=1)
    p.add_argument('--dt', type=float, default=1.0)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--visual-known-dist', type=float, default=None, help='Override visual memory new-cluster threshold')
    p.add_argument('--visual-norm-dist', type=float, default=None, help='Override visual novelty normalization distance')
    p.add_argument('--path-known-dist', type=float, default=None, help='Override path/context memory new-cluster threshold')
    p.add_argument('--path-norm-dist', type=float, default=None, help='Override path/context novelty normalization distance')
    return p.parse_args()


def main():
    args = parse_args()
    import cv2

    log_path = Path(args.log)
    frame_dir = Path(args.frame_dir)
    out_dir = Path(args.out_dir or (log_path.with_suffix('').as_posix() + f'_{args.backend}_replay'))
    out_frame_dir = out_dir / 'frames'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_frame_dir.mkdir(parents=True, exist_ok=True)
    out_log = out_dir / 'trajectory.jsonl'

    rows = load_rows(log_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    model = make_model(args.backend, action_dim=args.action_dim)
    if args.visual_known_dist is not None and hasattr(model, 'visual_memory'):
        model.visual_memory.known_dist = float(args.visual_known_dist)
    if args.path_known_dist is not None and hasattr(model, 'memory'):
        model.memory.known_dist = float(args.path_known_dist)
    if args.backend in ('pcvm-d', 'pcvm-j') and (args.visual_norm_dist is not None or args.path_norm_dist is not None):
        import VMM.pcvm_d as pcvm_d

        if args.visual_norm_dist is not None:
            pcvm_d.PCVM_VIS_MEMORY_NORM_DIST = float(args.visual_norm_dist)
        if args.path_norm_dist is not None:
            pcvm_d.PCVM_MEMORY_NORM_DIST = float(args.path_norm_dist)

    written = 0
    with out_log.open('w', buffering=1) as f:
        for i, row in enumerate(rows, 1):
            frame_path = find_frame(row, frame_dir)
            if frame_path is None:
                continue
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            sensors = row_sensors(row)
            action = row_action(row, args.action_dim)
            motion = np.array([0.0] + [float(x) for x in action], dtype=np.float32)
            result = model.observe(frame, sensors=sensors, motion=motion, action=action, dt=args.dt)
            backend = {k: v for k, v in result.items() if k != 'obs'}
            written += 1
            dst = out_frame_dir / f'step_{written:04d}.jpg'
            cv2.imwrite(str(dst), frame)
            rec = {
                'step': written,
                'source_step': row.get('step'),
                'source_frame': str(frame_path),
                'action': [float(x) for x in action],
                'distances': row.get('distances') or {},
                'backend': backend,
                'frame_path': str(dst),
            }
            f.write(json.dumps(rec, sort_keys=True) + '\n')
            print(json.dumps({
                'step': written,
                'visual_cluster': backend.get('pcvm_visual_cluster_id'),
                'path_cluster': backend.get('pcvm_path_cluster_id'),
                'visual_bank': backend.get('pcvm_visual_bank_size'),
                'path_bank': backend.get('pcvm_bank_size'),
                'visual_norm': backend.get('pcvm_visual_mem_norm'),
                'path_norm': backend.get('pcvm_path_mem_norm'),
            }, sort_keys=True), flush=True)

    sheets = make_cluster_sheets(out_log, out_frame_dir, out_dir)
    print(json.dumps({'backend': args.backend, 'replayed': written, 'saved_log': str(out_log), 'cluster_sheets': sheets}, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
