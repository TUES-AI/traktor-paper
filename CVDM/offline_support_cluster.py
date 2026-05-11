#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CVDM.checkpoint import _torch_load


def frame_path(run_dir: Path, record: dict[str, Any]) -> Path | None:
    rel = record.get("frame_tp1_path") or record.get("frame_t_path")
    if rel:
        path = run_dir / rel
        if path.exists():
            return path
    abs_path = record.get("frame_tp1_abs_path") or record.get("frame_t_abs_path")
    if abs_path and Path(abs_path).exists():
        return Path(abs_path)
    return None


def record_validity(record: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    motion = record.get("motion") or {}
    image_quality = record.get("image_quality_tp1") or {}
    distances = record.get("range_tp1") or {}
    reasons: list[str] = []

    if motion.get("contact_or_stall"):
        reasons.append("contact_or_stall")

    moved_cm = float(motion.get("executed_distance_cm") or 0.0)
    yaw_deg = abs(float(motion.get("executed_yaw_deg") or 0.0))
    if not (moved_cm >= args.visual_min_motion_cm or yaw_deg >= args.visual_min_yaw_deg):
        reasons.append("not_moved_or_rotated_enough")

    front = distances.get("front")
    left = distances.get("left")
    right = distances.get("right")
    if front is not None and float(front) < args.visual_min_front_cm:
        finite_ranges = [float(x) for x in (front, left, right) if x is not None]
        max_clear = max(finite_ranges) if finite_ranges else 0.0
        if max_clear < args.visual_front_close_clear_cm:
            reasons.append("front_close_no_open_range")

    if float(image_quality.get("laplacian_var") or 0.0) < args.image_min_laplacian_var:
        reasons.append("image_blur")
    if float(image_quality.get("mean") or 0.0) < args.image_min_mean:
        reasons.append("image_dark")
    if float(image_quality.get("mean") or 0.0) > args.image_max_mean:
        reasons.append("overbright_mean")
    if float(image_quality.get("std") or 0.0) < args.image_min_std:
        reasons.append("low_contrast")
    if float(image_quality.get("dark_frac") or 0.0) > args.image_max_dark_frac:
        reasons.append("too_many_dark_pixels")
    if float(image_quality.get("bright_frac") or 0.0) > args.image_max_bright_frac:
        reasons.append("too_many_bright_pixels")
    return not reasons, reasons


def load_cvdm_run(run_dir: Path, last_n: int) -> tuple[list[dict[str, Any]], np.ndarray]:
    transitions_path = run_dir / "transitions.json"
    checkpoint_path = run_dir / "models" / "cvdm_full.pt"
    if not transitions_path.exists():
        raise FileNotFoundError(f"missing transitions.json: {transitions_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing CVDM checkpoint with saved replay DINO vectors: {checkpoint_path}")

    records = json.loads(transitions_path.read_text())
    state = _torch_load(checkpoint_path, map_location="cpu")
    items = state["replay"]["items"]
    if last_n > 0:
        records = records[-last_n:]
        items = items[-last_n:]
    if len(records) != len(items):
        raise ValueError(f"record/replay mismatch: {len(records)} records vs {len(items)} replay items")
    vectors = np.stack([np.asarray(item.dino_tp1, dtype=np.float32) for item in items])
    return records, vectors


def support_vector_cluster(records: list[dict[str, Any]], vectors: np.ndarray, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    clusters: list[dict[str, Any]] = []
    deleted: list[dict[str, Any]] = []
    assignments: dict[int, dict[str, Any]] = {}

    for idx, (record, vector) in enumerate(zip(records, vectors)):
        ok, reasons = record_validity(record, args)
        if not ok:
            deleted.append({"idx": idx, "record": record, "reasons": reasons})
            continue

        best = (float("inf"), None, None)
        second = float("inf")
        for cluster_id, cluster in enumerate(clusters):
            for support_idx in cluster["supports"]:
                dist = float(np.linalg.norm(vector - vectors[support_idx]))
                if dist < best[0]:
                    second = best[0]
                    best = (dist, cluster_id, support_idx)
                elif dist < second:
                    second = dist

        if best[1] is None or best[0] > args.match_dist:
            cluster_id = len(clusters)
            clusters.append({
                "supports": [idx],
                "members": [idx],
                "created_step": int(record.get("step") or idx + 1),
            })
            assignments[idx] = {"cluster": cluster_id, "distance": None, "second": None, "new_cluster": True, "added_support": True}
            continue

        cluster = clusters[int(best[1])]
        cluster["members"].append(idx)
        add_support = best[0] > args.add_support_dist and len(cluster["supports"]) < args.max_supports
        if add_support:
            cluster["supports"].append(idx)
        assignments[idx] = {
            "cluster": int(best[1]),
            "distance": best[0],
            "second": None if not np.isfinite(second) else second,
            "new_cluster": False,
            "added_support": bool(add_support),
        }
    return clusters, deleted, assignments


def fonts():
    try:
        return (
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12),
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 11),
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16),
        )
    except Exception:
        font = ImageFont.load_default()
        return font, font, font


def render_cluster_sheets(
    run_dir: Path,
    out_dir: Path,
    records: list[dict[str, Any]],
    vectors: np.ndarray,
    clusters: list[dict[str, Any]],
    deleted: list[dict[str, Any]],
    assignments: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    font, small_font, title_font = fonts()
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999", "#66c2a5"]
    thumb_w, thumb_h = args.thumb_width, args.thumb_height
    pad = 6
    label_h = 34
    cols = args.cols
    row_h = thumb_h + label_h + pad

    def load_tile(idx: int, outline: str, support: bool = False) -> Image.Image:
        path = frame_path(run_dir, records[idx])
        try:
            image = Image.open(path).convert("RGB") if path is not None else Image.new("RGB", (thumb_w, thumb_h), "#cccccc")
            image.thumbnail((thumb_w, thumb_h))
            tile = Image.new("RGB", (thumb_w, thumb_h), "#eeeeee")
            tile.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))
        except Exception:
            tile = Image.new("RGB", (thumb_w, thumb_h), "#cccccc")
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, thumb_w - 1, thumb_h - 1), outline=outline, width=4 if support else 2)
        return tile

    left_w = 245
    overview_w = left_w + cols * thumb_w + (cols + 1) * pad
    overview_h = 48 + max(1, len(clusters)) * row_h + pad
    overview = Image.new("RGB", (overview_w, overview_h), "white")
    draw = ImageDraw.Draw(overview)
    draw.rectangle((0, 0, overview_w, 40), fill="#222222")
    draw.text((10, 10), f"Offline DINO support-vector banks: {len(clusters)} banks, valid={sum(len(c['members']) for c in clusters)}, deleted={len(deleted)}", fill="white", font=title_font)

    cluster_summaries: dict[str, Any] = {}
    y = 48
    for cluster_id, cluster in enumerate(clusters):
        color = colors[cluster_id % len(colors)]
        members = cluster["members"]
        supports = set(cluster["supports"])
        rewards = [float(records[i].get("reward") or 0.0) for i in members]
        steps = [int(records[i].get("step") or i + 1) for i in members]
        distances = [float((records[i].get("motion") or {}).get("executed_distance_cm") or 0.0) for i in members]
        yaws = [abs(float((records[i].get("motion") or {}).get("executed_yaw_deg") or 0.0)) for i in members]
        cluster_summaries[str(cluster_id)] = {
            "count": len(members),
            "supports": list(cluster["supports"]),
            "support_steps": [int(records[i].get("step") or i + 1) for i in cluster["supports"]],
            "step_range": [min(steps), max(steps)],
            "reward_sum": sum(rewards),
            "mean_reward": sum(rewards) / len(rewards),
            "mean_distance_cm": sum(distances) / len(distances),
            "mean_abs_yaw_deg": sum(yaws) / len(yaws),
            "created_step": cluster["created_step"],
        }

        draw.rectangle((10, y, 24, y + row_h - pad), fill=color)
        draw.text((34, y + 2), f"bank {cluster_id}", fill="black", font=title_font)
        draw.text((34, y + 23), f"n={len(members)}  supports={len(supports)}", fill="black", font=font)
        draw.text((34, y + 40), f"steps {min(steps)}-{max(steps)}", fill="black", font=font)
        draw.text((34, y + 57), f"reward {sum(rewards):+.2f}", fill="black", font=font)
        draw.text((34, y + 74), f"mean yaw {sum(yaws) / len(yaws):.0f}°", fill="black", font=font)

        picks = [members[round(k * (len(members) - 1) / (cols - 1))] for k in range(cols)] if len(members) > cols else list(members)
        forced = list(cluster["supports"][: min(cols, len(cluster["supports"]))])
        picks = (forced + [p for p in picks if p not in forced])[:cols]
        x0 = left_w + pad
        for j, idx in enumerate(picks):
            x = x0 + j * (thumb_w + pad)
            support = idx in supports
            overview.paste(load_tile(idx, color, support=support), (x, y))
            record = records[idx]
            assignment = assignments.get(idx, {})
            label = "S" if support else "m"
            draw.text((x, y + thumb_h + 2), f"{label} s{int(record.get('step') or idx + 1):03d} r{float(record.get('reward') or 0):+.2f}", fill="black", font=small_font)
            dist = assignment.get("distance")
            draw.text((x, y + thumb_h + 17), "new" if dist is None else f"d={dist:.2f}", fill="black", font=small_font)
        y += row_h

    overview_path = out_dir / "support_vector_banks_overview.jpg"
    overview.save(overview_path, quality=92)

    bank_sheets: dict[str, str] = {}
    for cluster_id, cluster in enumerate(clusters):
        color = colors[cluster_id % len(colors)]
        members = cluster["members"]
        supports = set(cluster["supports"])
        max_tiles = min(args.max_bank_tiles, len(members))
        picks = [members[round(k * (len(members) - 1) / (max_tiles - 1))] for k in range(max_tiles)] if len(members) > max_tiles and max_tiles > 1 else list(members[:max_tiles])
        forced = list(cluster["supports"])
        picks = (forced + [p for p in picks if p not in forced])[:max_tiles]
        rows = max(1, (len(picks) + cols - 1) // cols)
        width = cols * thumb_w + (cols + 1) * pad
        height = 42 + rows * row_h + pad
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width, 38), fill=color)
        summary = cluster_summaries[str(cluster_id)]
        draw.text((10, 9), f"support-vector bank {cluster_id}: n={summary['count']}, supports={len(cluster['supports'])}, reward={summary['reward_sum']:+.2f}", fill="white", font=title_font)
        for j, idx in enumerate(picks):
            x = pad + (j % cols) * (thumb_w + pad)
            yy = 42 + pad + (j // cols) * row_h
            support = idx in supports
            image.paste(load_tile(idx, color, support=support), (x, yy))
            record = records[idx]
            assignment = assignments.get(idx, {})
            label = "SUPPORT" if support else "member"
            draw.text((x, yy + thumb_h + 2), f"{label} s{int(record.get('step') or idx + 1):03d} r{float(record.get('reward') or 0):+.2f}", fill="black", font=small_font)
            dist = assignment.get("distance")
            draw.text((x, yy + thumb_h + 17), "new cluster" if dist is None else f"nearest support d={dist:.2f}", fill="black", font=small_font)
        path = out_dir / f"support_vector_bank_{cluster_id:02d}.jpg"
        image.save(path, quality=92)
        bank_sheets[str(cluster_id)] = str(path)

    reason_counts = Counter()
    for item in deleted:
        for reason in item["reasons"]:
            reason_counts[reason] += 1

    deleted_path = None
    if deleted:
        sample_n = min(args.max_deleted_tiles, len(deleted))
        picks = [deleted[round(k * (len(deleted) - 1) / (sample_n - 1))] for k in range(sample_n)] if len(deleted) > 1 else list(deleted)
        rows = max(1, (len(picks) + cols - 1) // cols)
        width = cols * thumb_w + (cols + 1) * pad
        height = 46 + rows * (thumb_h + 46 + pad) + pad
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width, 42), fill="#555555")
        draw.text((10, 10), f"deleted by gates: {len(deleted)} frames, sample {len(picks)}", fill="white", font=title_font)
        for j, item in enumerate(picks):
            idx = item["idx"]
            x = pad + (j % cols) * (thumb_w + pad)
            yy = 46 + pad + (j // cols) * (thumb_h + 46 + pad)
            image.paste(load_tile(idx, "#555555", support=False), (x, yy))
            record = records[idx]
            reason_text = ",".join(item["reasons"])
            draw.text((x, yy + thumb_h + 2), f"s{int(record.get('step') or idx + 1):03d} r{float(record.get('reward') or 0):+.2f}", fill="black", font=small_font)
            draw.text((x, yy + thumb_h + 17), reason_text[:27], fill="black", font=small_font)
            draw.text((x, yy + thumb_h + 32), reason_text[27:54], fill="black", font=small_font)
        deleted_path = out_dir / "support_vector_deleted_frames.jpg"
        image.save(deleted_path, quality=92)

    return {
        "clusters": cluster_summaries,
        "deleted_reasons": dict(reason_counts.most_common()),
        "overview": str(overview_path),
        "deleted_sheet": None if deleted_path is None else str(deleted_path),
        "bank_sheets": bank_sheets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline support-vector DINO clustering for saved CVDM rover runs.")
    parser.add_argument("--run-dir", required=True, type=Path, help="CVDM run directory containing transitions.json and models/cvdm_full.pt")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--last-n", type=int, default=100)
    parser.add_argument("--match-dist", type=float, default=1.00)
    parser.add_argument("--add-support-dist", type=float, default=0.45)
    parser.add_argument("--max-supports", type=int, default=8)

    parser.add_argument("--visual-min-motion-cm", type=float, default=10.0)
    parser.add_argument("--visual-min-yaw-deg", type=float, default=5.0)
    parser.add_argument("--visual-min-front-cm", type=float, default=25.0)
    parser.add_argument("--visual-front-close-clear-cm", type=float, default=100.0)
    parser.add_argument("--image-min-laplacian-var", type=float, default=25.0)
    parser.add_argument("--image-min-mean", type=float, default=18.0)
    parser.add_argument("--image-max-mean", type=float, default=238.0)
    parser.add_argument("--image-min-std", type=float, default=8.0)
    parser.add_argument("--image-max-dark-frac", type=float, default=0.55)
    parser.add_argument("--image-max-bright-frac", type=float, default=0.45)

    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--thumb-width", type=int, default=150)
    parser.add_argument("--thumb-height", type=int, default=112)
    parser.add_argument("--max-bank-tiles", type=int, default=24)
    parser.add_argument("--max-deleted-tiles", type=int, default=32)
    parser.add_argument("--open", action="store_true", help="Open overview/deleted sheets on macOS after rendering")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    out_dir = args.out_dir or (run_dir / "offline_support_banks")
    records, vectors = load_cvdm_run(run_dir, args.last_n)
    clusters, deleted, assignments = support_vector_cluster(records, vectors, args)
    rendered = render_cluster_sheets(run_dir, out_dir, records, vectors, clusters, deleted, assignments, args)

    summary = {
        "source_run": str(run_dir),
        "algorithm": "online nearest-exemplar support-vector clustering on saved DINOv3 embeddings",
        "match_dist": args.match_dist,
        "add_support_dist": args.add_support_dist,
        "max_supports": args.max_supports,
        "records": len(records),
        "valid_frames": sum(len(c["members"]) for c in clusters),
        "deleted_frames": len(deleted),
        "thresholds": {
            "visual_min_motion_cm": args.visual_min_motion_cm,
            "visual_min_yaw_deg": args.visual_min_yaw_deg,
            "visual_min_front_cm": args.visual_min_front_cm,
            "visual_front_close_clear_cm": args.visual_front_close_clear_cm,
            "image_min_laplacian_var": args.image_min_laplacian_var,
            "image_min_mean": args.image_min_mean,
            "image_max_mean": args.image_max_mean,
            "image_min_std": args.image_min_std,
            "image_max_dark_frac": args.image_max_dark_frac,
            "image_max_bright_frac": args.image_max_bright_frac,
        },
        **rendered,
    }
    summary_path = out_dir / "support_vector_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({**summary, "summary": str(summary_path)}, indent=2, sort_keys=True))

    if args.open:
        for key in ("overview", "deleted_sheet"):
            path = summary.get(key)
            if path:
                subprocess.run(["open", path], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
