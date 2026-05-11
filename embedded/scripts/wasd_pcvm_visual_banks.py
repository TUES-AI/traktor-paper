#!/usr/bin/env python3
"""WASD-drive the rover while logging PCVM visual/path memory clusters.

Default backend is PCVM-D (DINOv2). This is a diagnostic script, not an RL run:
you manually drive through rooms/corridors, it logs camera frames + PCVM memory
IDs and writes cluster-contact-sheet PNGs on exit.

Keys:
- w/s: forward/backward
- a/d: spin left/right
- q/e: forward-left / forward-right arc
- x or space: stop
- p: capture/log one frame while stopped
- ESC: quit
"""

import argparse
import json
from pathlib import Path
from select import select
import sys
import termios
import time
import tty

import numpy as np

import _paths  # noqa: F401
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from VMM.pcvm import clamp


ULTRA_MAX_CM = 400.0


def norm_distance(cm):
    if cm is None:
        return 1.0
    return clamp(float(cm) / ULTRA_MAX_CM, 0.0, 1.0)


def set_raw_terminal():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(fd)
    return fd, old_settings


def read_pending_key(timeout=0.02):
    ready, _, _ = select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    key = sys.stdin.read(1)
    while select([sys.stdin], [], [], 0)[0]:
        key = sys.stdin.read(1)
    if key == '\x1b':
        return 'escape'
    return key.lower()


def make_model(backend, action_dim, dino_model_name='facebook/dinov2-small'):
    if backend in ('pcvm-d', 'pcvm-j'):
        from VMM.pcvm_d import PCVMDINO

        return PCVMDINO(action_dim=action_dim, model_name=dino_model_name)
    if backend == 'pcvm-m':
        from VMM.pcvm_m import PCVMMobileNet

        return PCVMMobileNet(action_dim=action_dim)
    if backend == 'pcvm':
        from VMM.pcvm import PCVM

        return PCVM(action_dim=action_dim)
    raise ValueError(f'unknown backend: {backend}')


def command_for_key(key, speed, arc_scale):
    if key == 'w':
        return 'forward', 'forward', speed, speed, 'forward', np.array([0.0, 1.0], dtype=np.float32)
    if key == 's':
        return 'backward', 'backward', speed, speed, 'backward', np.array([0.0, -1.0], dtype=np.float32)
    if key == 'a':
        return 'backward', 'forward', speed, speed, 'spin_left', np.array([-1.0, 0.0], dtype=np.float32)
    if key == 'd':
        return 'forward', 'backward', speed, speed, 'spin_right', np.array([1.0, 0.0], dtype=np.float32)
    if key == 'q':
        return 'forward', 'forward', speed * arc_scale, speed, 'arc_left', np.array([-0.6, 1.0], dtype=np.float32)
    if key == 'e':
        return 'forward', 'forward', speed, speed * arc_scale, 'arc_right', np.array([0.6, 1.0], dtype=np.float32)
    return None


def read_distances(rover):
    raw = rover.get_ultrasonic(timeout_seconds=0.03)
    return {'right': raw.get(1), 'left': raw.get(2), 'front': raw.get(3)}


def save_frame(frame, path):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)


def make_cluster_sheets(log_path, frame_dir, out_dir, keys=('pcvm_visual_cluster_id', 'pcvm_path_cluster_id')):
    from collections import defaultdict
    from PIL import Image, ImageDraw

    rows = []
    for line in Path(log_path).read_text(errors='ignore').splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    outputs = []
    for key in keys:
        groups = defaultdict(list)
        for row in rows:
            backend = row.get('backend') or {}
            cid = backend.get(key)
            if cid is None:
                continue
            groups[cid].append(row)
        if not groups:
            continue
        thumb_w, thumb_h = 160, 120
        cols = 10
        label_w = 220
        row_h = thumb_h + 42
        width = label_w + cols * (thumb_w + 8) + 20
        height = 48 + len(groups) * row_h
        canvas = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), f'{Path(log_path).stem} grouped by {key}', fill='black')
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999']
        y = 40
        for idx, cid in enumerate(sorted(groups, key=lambda x: int(x))):
            items = groups[cid]
            if len(items) > cols:
                picks = [items[round(i * (len(items) - 1) / (cols - 1))] for i in range(cols)]
            else:
                picks = items
            color = colors[idx % len(colors)]
            steps = [int(r.get('step') or 0) for r in items]
            draw.rectangle((10, y, 20, y + row_h - 8), fill=color)
            draw.text((28, y + 4), f'cluster {cid}', fill='black')
            draw.text((28, y + 24), f'n={len(items)} steps {min(steps)}-{max(steps)}', fill='black')
            for j, row in enumerate(picks):
                step = int(row.get('step') or 0)
                fp = frame_dir / f'step_{step:04d}.jpg'
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
                b = row.get('backend') or {}
                draw.text((x, y + thumb_h + 2), f's{step:03d} v{float(b.get("pcvm_visual_mem_norm") or 0):.2f}', fill='black')
                draw.text((x, y + thumb_h + 18), f'p{float(b.get("pcvm_path_mem_norm") or 0):.2f}', fill='black')
            y += row_h
        out = out_dir / f'{Path(log_path).stem}_{key}_clusters.png'
        canvas.save(out)
        outputs.append(str(out))
    return outputs


def parse_args():
    p = argparse.ArgumentParser(description='Manual WASD PCVM visual-bank logger.')
    p.add_argument('--backend', choices=['pcvm-d', 'pcvm-j', 'pcvm-m', 'pcvm'], default='pcvm-d')
    p.add_argument('--dino-model-name', default='facebook/dinov2-small')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--run-name', default=None)
    p.add_argument('--speed', type=float, default=55.0)
    p.add_argument('--arc-scale', type=float, default=0.45)
    p.add_argument('--stop-after-seconds', type=float, default=0.25)
    p.add_argument('--observe-interval', type=float, default=0.75)
    return p.parse_args()


def main():
    from api.rover_api import RoverAPI

    args = parse_args()
    run_name = args.run_name or f'{args.backend}_manual_banks_{time.strftime("%Y%m%d_%H%M%S")}'
    out_dir = Path(args.out_dir or 'results') / run_name
    frame_dir = out_dir / 'frames'
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'trajectory.jsonl'

    print(__doc__.strip(), flush=True)
    print(json.dumps({'backend': args.backend, 'out_dir': str(out_dir), 'speed': args.speed}, sort_keys=True), flush=True)
    print('Loading visual model; first run may download DINOv2 weights.', flush=True)
    model = make_model(args.backend, action_dim=2, dino_model_name=args.dino_model_name)
    rover = RoverAPI(camera_enabled=True)
    fd = None
    old = None
    active_key = None
    last_key_time = 0.0
    last_obs_time = 0.0
    last_action = np.zeros(2, dtype=np.float32)
    step = 0
    log_f = log_path.open('a', buffering=1)
    try:
        fd, old = set_raw_terminal()
        rover.stop_motors()
        while True:
            now = time.monotonic()
            key = read_pending_key()
            if active_key and now - last_key_time > args.stop_after_seconds:
                rover.stop_motors()
                active_key = None
                last_action = np.zeros(2, dtype=np.float32)

            force_observe = False
            if key is not None:
                if key == 'escape':
                    break
                if key in ('x', ' '):
                    rover.stop_motors()
                    active_key = None
                    last_action = np.zeros(2, dtype=np.float32)
                elif key == 'p':
                    force_observe = True
                else:
                    cmd = command_for_key(key, args.speed, args.arc_scale)
                    if cmd is not None:
                        left_dir, right_dir, left_speed, right_speed, label, action = cmd
                        rover.drive(left_dir, right_dir, left_speed=left_speed, right_speed=right_speed)
                        active_key = key
                        last_key_time = now
                        last_action = action
                        print(label, flush=True)

            if force_observe or (now - last_obs_time >= args.observe_interval):
                dt = max(1e-3, now - last_obs_time) if last_obs_time > 0.0 else args.observe_interval
                last_obs_time = now
                step += 1
                distances = read_distances(rover)
                sensors = np.array([norm_distance(distances['left']), norm_distance(distances['right']), norm_distance(distances['front'])], dtype=np.float32)
                motion = np.array([0.0, float(last_action[0]), float(last_action[1])], dtype=np.float32)
                frame = rover.get_camera_frame()
                frame_path = frame_dir / f'step_{step:04d}.jpg'
                save_frame(frame, frame_path)
                result = model.observe(frame, sensors=sensors, motion=motion, action=last_action, dt=dt)
                backend = {k: v for k, v in result.items() if k != 'obs'}
                record = {
                    'step': step,
                    'time': time.time(),
                    'active_key': active_key,
                    'action': [float(x) for x in last_action],
                    'distances': distances,
                    'backend': backend,
                    'frame_path': str(frame_path),
                }
                log_f.write(json.dumps(record, sort_keys=True) + '\n')
                print(json.dumps({
                    'step': step,
                    'visual_cluster': backend.get('pcvm_visual_cluster_id'),
                    'path_cluster': backend.get('pcvm_path_cluster_id'),
                    'visual_bank': backend.get('pcvm_visual_bank_size'),
                    'path_bank': backend.get('pcvm_bank_size'),
                    'visual_norm': backend.get('pcvm_visual_mem_norm'),
                    'path_norm': backend.get('pcvm_path_mem_norm'),
                }, sort_keys=True), flush=True)
    finally:
        rover.stop_motors()
        rover.close()
        log_f.close()
        if fd is not None and old is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sheets = make_cluster_sheets(log_path, frame_dir, out_dir)
        print(json.dumps({'saved_log': str(log_path), 'cluster_sheets': sheets}, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
