#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EMBEDDED_ROOT = REPO_ROOT / "embedded"
for p in (REPO_ROOT, EMBEDDED_ROOT, EMBEDDED_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from CVDM.checkpoint import load_cvdm_checkpoint, save_cvdm_checkpoint, save_individual_model_params
from CVDM.config import CVDMConfig
from CVDM.logger import CVDMRunLogger, to_jsonable, utc_now_iso
from CVDM.observation import normalized_ranges
from CVDM.replay import CVDMTransition, TransitionReplayBuffer
from CVDM.reward import reward_from_transition
from CVDM.training import CVDMTrainer
from CVDM.vision import make_visual_encoder


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def finite_front(front: Any) -> float:
    if front is None:
        return 0.0
    return float(front)


class RealCVDMSACEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        from api.rover_api import RoverAPI
        from control.local_target_executor import LocalTargetExecutor, LocalTargetExecutorConfig
        from control.safety import SafetyConfig, SafetyController
        from drivers.sensors.mpu9150 import MPU9150

        self.args = args
        self.config = CVDMConfig(
            lr=args.cvdm_lr,
            rnd_lr=args.cvdm_rnd_lr,
            memory_size=args.cvdm_memory_size,
            memory_known_distance=args.cvdm_memory_known_distance,
            memory_norm_distance=args.cvdm_memory_norm_distance,
            memory_min_assignment_margin=args.cvdm_memory_min_assignment_margin,
            visual_min_motion_cm=args.visual_min_motion_cm,
            visual_min_yaw_deg=args.visual_min_yaw_deg,
            visual_min_front_cm=args.visual_min_front_cm,
            visual_front_close_clear_cm=args.visual_front_close_clear_cm,
            image_min_laplacian_var=args.image_min_laplacian_var,
            image_min_mean=args.image_min_mean,
            image_max_mean=args.image_max_mean,
            image_min_std=args.image_min_std,
            image_max_dark_frac=args.image_max_dark_frac,
            image_max_bright_frac=args.image_max_bright_frac,
            none_range_norm=1.0 if args.no_echo_is_clear else 0.0,
            safe_front_min_cm=args.safe_front_min_cm,
            safe_motion_min_cm=args.safe_motion_min_cm,
            novelty_weight=args.cvdm_novelty_weight,
            learning_progress_weight=args.cvdm_learning_progress_weight,
            distance_reward_weight=args.cvdm_distance_reward_weight,
            safe_motion_bonus=args.cvdm_safe_motion_bonus,
            new_cluster_bonus=args.cvdm_new_cluster_bonus,
            contact_penalty=args.cvdm_contact_penalty,
            zero_progress_penalty=args.cvdm_zero_progress_penalty,
            recovery_penalty=args.cvdm_recovery_penalty,
            near_obstacle_penalty=args.cvdm_near_obstacle_penalty,
            obstructed_forward_penalty=args.cvdm_obstructed_forward_penalty,
            obstructed_forward_front_cm=args.cvdm_obstructed_forward_front_cm,
            obstructed_forward_theta_deg=args.cvdm_obstructed_forward_theta_deg,
            clear_front_turn_penalty=args.cvdm_clear_front_turn_penalty,
            clear_front_turn_start_cm=args.cvdm_clear_front_turn_start_cm,
            clear_front_turn_scale_cm=args.cvdm_clear_front_turn_scale_cm,
            metadata={"script": "CVDM/train_real_rover.py"},
        )
        self.logger = CVDMRunLogger(args.out_dir)
        self.visual_encoder = make_visual_encoder(args.visual_encoder, input_size=args.dino_input_size, threads=args.dino_threads)
        self.trainer = CVDMTrainer(self.config, device="cpu")
        self.replay = TransitionReplayBuffer(maxlen=args.cvdm_replay_size)
        self.resume_state: dict[str, Any] | None = None
        self._load_resume_cvdm_if_requested()

        obs_dim = self.config.phi_dim + self.config.range_dim + self.config.action_dim + len(self.config.candidate_actions) * 2 + 2
        self.action_space = spaces.Box(low=np.array([-1.0], dtype=np.float32), high=np.array([1.0], dtype=np.float32))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.rover = RoverAPI(camera_enabled=True)
        self.imu = MPU9150(bus=1, address=0x68)
        self.safety = SafetyController(
            self.rover,
            imu=self.imu,
            config=SafetyConfig(
                min_front_stop_cm=args.front_stop_cm,
                max_front_stop_cm=args.front_stop_cm,
                front_clear_to_resume_cm=args.front_clear_cm,
                no_echo_is_clear=args.no_echo_is_clear,
            ),
        )
        self.executor = LocalTargetExecutor(
            self.safety,
            config=LocalTargetExecutorConfig(
                turn_pwm=args.turn_pwm,
                drive_pwm=args.drive_pwm,
                until_front_stop_cm=args.until_front_cm,
                max_drive_seconds=args.until_front_max_seconds,
                cm_per_second=args.cm_per_second,
            ),
            status_callback=lambda s: print(json.dumps({"status": s}, sort_keys=True), flush=True),
        )
        self.step_count = 0
        self.current_state: dict[str, Any] | None = None
        self.last_executed_action = np.zeros(1, dtype=np.float32)
        self.last_transition_surprise = 0.0
        self.start_wall_time = utc_now_iso()
        self.gyro_bias = self.safety.calibrate_gyro()
        self._write_initial_manifest()

    def _shape_policy_theta(self, raw_theta_norm: float) -> float:
        raw = clamp(raw_theta_norm, -1.0, 1.0)
        eps = float(self.args.theta_deadzone)
        gamma = float(self.args.theta_power_gamma)
        if abs(raw) < eps:
            return 0.0
        x = math.copysign((abs(raw) - eps) / max(1e-6, 1.0 - eps), raw)
        return clamp(math.copysign(abs(x) ** gamma, x), -1.0, 1.0)

    def _write_initial_manifest(self) -> None:
        visual_meta = self.visual_encoder.metadata() if hasattr(self.visual_encoder, "metadata") else {"kind": self.args.visual_encoder}
        self.logger.write_manifest(
            {
                "run_name": self.args.run_name,
                "created_utc": self.start_wall_time,
                "out_dir": str(self.logger.out_dir),
                "steps_requested": self.args.steps,
                "visual_encoder": visual_meta,
                "resume": self.resume_state,
                "cvdm_config": self.config.to_dict(),
                "gyro_z_bias": self.gyro_bias,
                "args": vars(self.args),
            }
        )

    def _resume_path(self, explicit: str | None, default_relative: str) -> Path | None:
        if explicit:
            return Path(explicit)
        if not self.args.resume_dir:
            return None
        return Path(self.args.resume_dir) / default_relative

    def _load_resume_cvdm_if_requested(self) -> None:
        cvdm_path = self._resume_path(self.args.resume_cvdm, "models/cvdm_full.pt")
        if cvdm_path is None:
            return
        if not cvdm_path.exists():
            raise FileNotFoundError(f"CVDM resume checkpoint not found: {cvdm_path}")
        state = load_cvdm_checkpoint(
            self.trainer,
            cvdm_path,
            replay=None if self.args.no_resume_replay else self.replay,
            map_location=self.trainer.device,
        )
        self.resume_state = {
            "cvdm_path": str(cvdm_path),
            "cvdm_step": int(self.trainer.step),
            "cvdm_replay_size": len(self.replay),
            "checkpoint_format": state.get("format"),
            "checkpoint_extra": state.get("extra", {}),
        }
        print(json.dumps({"resume_cvdm": self.resume_state}, sort_keys=True), flush=True)

    def close(self) -> None:
        try:
            self.rover.stop_motors()
        except Exception:
            pass
        for obj in (getattr(self, "safety", None), getattr(self, "imu", None), getattr(self, "rover", None)):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass

    def _read_imu_safe(self) -> dict[str, Any] | None:
        try:
            return self.imu.read_all()
        except Exception as exc:
            return {"error": repr(exc)}

    def _image_quality(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        import cv2

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean = float(gray.mean())
        std = float(gray.std())
        dark_frac = float((gray < 10).mean())
        bright_frac = float((gray > 245).mean())
        reasons = []
        if blur_var < self.config.image_min_laplacian_var:
            reasons.append("blur")
        if mean < self.config.image_min_mean:
            reasons.append("dark")
        if mean > self.config.image_max_mean:
            reasons.append("overbright_mean")
        if std < self.config.image_min_std:
            reasons.append("low_contrast")
        if dark_frac > self.config.image_max_dark_frac:
            reasons.append("too_many_dark_pixels")
        if bright_frac > self.config.image_max_bright_frac:
            reasons.append("too_many_bright_pixels")
        return {
            "ok": not reasons,
            "reasons": reasons,
            "laplacian_var": blur_var,
            "mean": mean,
            "std": std,
            "dark_frac": dark_frac,
            "bright_frac": bright_frac,
        }

    def _capture_observation(self, label: str, last_action: np.ndarray) -> dict[str, Any]:
        t0 = time.monotonic()
        wall = utc_now_iso()
        distances = self.safety.read_distances()
        imu = self._read_imu_safe()
        frame = self.rover.get_camera_frame()
        image_quality = self._image_quality(frame)
        frame_ref = self.logger.save_frame(frame, label)
        dino = self.visual_encoder.encode(frame)
        ranges = normalized_ranges(distances, self.config)
        obs, obs_summary = self.trainer.policy_observation(
            dino,
            ranges,
            np.asarray(last_action, dtype=np.float32).reshape(1),
            distances,
            transition_surprise=self.last_transition_surprise,
        )
        return {
            "wall_time_utc": wall,
            "monotonic_time": t0,
            "distances": distances,
            "ranges_norm": ranges,
            "imu": imu,
            "frame": frame_ref,
            "image_quality": image_quality,
            "dino": dino,
            "obs": obs,
            "obs_summary": obs_summary,
            "last_action": np.asarray(last_action, dtype=np.float32).reshape(1),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.last_executed_action = np.zeros(1, dtype=np.float32)
        self.last_transition_surprise = 0.0
        self.current_state = self._capture_observation("reset", self.last_executed_action)
        return self.current_state["obs"].astype(np.float32), {"distances": self.current_state["distances"]}

    def _recover_if_blocked(self, distances: dict) -> dict[str, Any] | None:
        reverse = self.safety.reverse_if_too_close(self.args.drive_pwm, distances)
        if reverse is not None:
            distances = self.safety.read_distances()
        safe, front, threshold = self.safety.is_front_safe(self.args.drive_pwm, distances)
        if safe:
            return {"reverse": reverse, "continue_after_reverse": True} if reverse is not None else None
        turn_dir = self.safety.freer_side(distances)
        report = self.safety.turn_until_clear(turn_dir, speed_pct=self.args.turn_pwm)
        return {
            "reverse": reverse,
            "front_cm": front,
            "threshold_cm": threshold,
            "turn_dir": turn_dir,
            "recovery": report,
            "continue_after_reverse": False,
        }

    def _executed_action_from_feedback(self, execution: dict | None, recovery: dict | None) -> np.ndarray:
        theta_deg = 0.0
        if recovery:
            report = recovery.get("recovery") or {}
            theta_deg += float(report.get("yaw_deg") or 0.0)
        if execution:
            turn = execution.get("turn") or {}
            theta_deg += float(turn.get("yaw_deg") or execution.get("theta_deg") or 0.0)
        theta_norm = clamp(theta_deg / max(1e-6, self.args.max_theta_deg), -1.0, 1.0)
        return np.array([theta_norm], dtype=np.float32)

    def _motion_terms(self, execution: dict | None, recovery: dict | None, post_distances: dict, executed_action: np.ndarray) -> dict[str, Any]:
        execution = execution or {}
        drive = execution.get("drive") or {}
        distance_cm = max(0.0, float(execution.get("clipped_distance_cm") or drive.get("estimated_distance_cm") or 0.0))
        if drive and not drive.get("ok"):
            distance_cm = 0.0 if drive.get("reason") == "contact_or_stall" else distance_cm
        reason_text = f"{execution.get('reason') or ''} {drive.get('reason') or ''}"
        contact = bool(drive.get("contact_or_stall") or "contact_or_stall" in reason_text)
        recovery_event = bool(recovery) or bool(execution.get("reverse_recovery")) or bool(drive.get("close_front_recovery"))
        front_after = post_distances.get("front")
        executed_yaw_deg = float(np.asarray(executed_action, dtype=np.float32).reshape(-1)[0]) * self.args.max_theta_deg
        visual_motion_valid = distance_cm >= self.config.visual_min_motion_cm or abs(executed_yaw_deg) >= self.config.visual_min_yaw_deg
        zero_progress = not visual_motion_valid
        return {
            "executed_distance_cm": float(distance_cm),
            "executed_yaw_deg": float(executed_yaw_deg),
            "visual_motion_valid": bool(visual_motion_valid),
            "front_after_cm": None if front_after is None else float(front_after),
            "contact_or_stall": bool(contact),
            "recovery": bool(recovery_event),
            "zero_progress": bool(zero_progress),
        }

    def _visual_wall_gate(self, distances: dict) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        front = distances.get("front")
        left = distances.get("left")
        right = distances.get("right")
        if front is not None and float(front) < self.config.visual_min_front_cm:
            finite_ranges = [float(x) for x in (front, left, right) if x is not None]
            max_clear = max(finite_ranges) if finite_ranges else 0.0
            if max_clear < self.config.visual_front_close_clear_cm:
                reasons.append("front_close_no_open_range")
        return not reasons, reasons

    def _memory_update_gate(self, motion: dict[str, Any], post: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        reasons: list[str] = []
        if motion["contact_or_stall"]:
            reasons.append("contact_or_stall")
        if not motion["visual_motion_valid"]:
            reasons.append("not_moved_or_rotated_enough")
        wall_ok, wall_reasons = self._visual_wall_gate(post["distances"])
        if not wall_ok:
            reasons.extend(wall_reasons)
        image_quality = post.get("image_quality") or {}
        if not image_quality.get("ok", False):
            reasons.extend(["image_" + str(x) for x in image_quality.get("reasons", [])])
        ok = not reasons
        detail = {
            "ok": ok,
            "reasons": reasons,
            "visual_motion_valid": bool(motion["visual_motion_valid"]),
            "wall_ok": wall_ok,
            "image_quality_ok": bool(image_quality.get("ok", False)),
        }
        return ok, "ok" if ok else ",".join(reasons), detail

    def step(self, action):
        if self.current_state is None:
            self.current_state = self._capture_observation("autoreset", self.last_executed_action)
        pre = self.current_state
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        theta_raw = clamp(float(action[0]), -1.0, 1.0)
        theta_norm = self._shape_policy_theta(theta_raw)
        theta_deg = theta_norm * self.args.max_theta_deg
        action_start = time.monotonic()
        recovery = self._recover_if_blocked(pre["distances"])
        execution = None
        target = {"mode": "theta_until_front", "theta_norm": theta_norm, "theta_deg": theta_deg, "front_stop_cm": self.args.until_front_cm}
        if recovery is None or recovery.get("continue_after_reverse"):
            execution = self.executor.execute_theta_until_front(theta_deg, self.args.until_front_cm)
        action_end = time.monotonic()
        if self.args.settle_seconds > 0:
            time.sleep(self.args.settle_seconds)
        executed_action = self._executed_action_from_feedback(execution, recovery)
        post = self._capture_observation(f"step_{self.step_count:04d}_post", executed_action)
        motion = self._motion_terms(execution, recovery, post["distances"], executed_action)
        memory_update_ok, memory_update_reason, memory_update_detail = self._memory_update_gate(motion, post)

        transition = CVDMTransition(
            dino_t=pre["dino"],
            range_t=pre["ranges_norm"],
            last_action_t=pre["last_action"],
            action_executed=executed_action,
            dino_tp1=post["dino"],
            range_tp1=post["ranges_norm"],
            executed_distance_cm=motion["executed_distance_cm"],
            front_after_cm=finite_front(motion["front_after_cm"]),
            contact_or_stall=motion["contact_or_stall"],
            recovery=motion["recovery"],
            metadata={
                "step": self.step_count,
                "memory_update_ok": bool(memory_update_ok),
                "memory_update_reason": memory_update_reason,
                "memory_update_detail": memory_update_detail,
            },
        )
        cvdm_metrics = self.trainer.observe_transition(
            transition,
            self.replay,
            batch_size=self.args.cvdm_batch_size,
            gradient_steps=self.args.cvdm_gradient_steps,
        )
        self.last_transition_surprise = float(cvdm_metrics.get("forward_error_after") or 0.0)
        post_obs, post_summary = self.trainer.policy_observation(
            post["dino"], post["ranges_norm"], executed_action, post["distances"], transition_surprise=self.last_transition_surprise
        )
        post["obs"] = post_obs
        post["obs_summary"] = post_summary

        reward, reward_terms = reward_from_transition(
            self.config,
            novelty=float(cvdm_metrics.get("novelty_phi") or 0.0),
            learning_progress=float(cvdm_metrics.get("learning_progress") or 0.0),
            executed_distance_cm=motion["executed_distance_cm"],
            front_after_cm=motion["front_after_cm"],
            contact_or_stall=motion["contact_or_stall"],
            recovery=motion["recovery"],
            zero_progress=motion["zero_progress"],
            new_cluster=bool(cvdm_metrics.get("density_new_cluster")),
            visual_memory_valid=bool(memory_update_ok),
            pre_front_cm=pre["distances"].get("front"),
            requested_theta_deg=theta_deg,
            requested_theta_norm=theta_norm,
        )
        record = {
            "step": self.step_count,
            "timestamp_utc": utc_now_iso(),
            "action_start_monotonic": action_start,
            "action_end_monotonic": action_end,
            "duration_action_s": action_end - action_start,
            "frame_t_path": pre["frame"]["relative_path"],
            "frame_t_abs_path": pre["frame"]["path"],
            "frame_tp1_path": post["frame"]["relative_path"],
            "frame_tp1_abs_path": post["frame"]["path"],
            "time_t": {"wall_time_utc": pre["wall_time_utc"], "monotonic": pre["monotonic_time"]},
            "time_tp1": {"wall_time_utc": post["wall_time_utc"], "monotonic": post["monotonic_time"]},
            "range_t": pre["distances"],
            "range_tp1": post["distances"],
            "range_t_norm": pre["ranges_norm"],
            "range_tp1_norm": post["ranges_norm"],
            "imu_t": pre["imu"],
            "imu_tp1": post["imu"],
            "image_quality_t": pre["image_quality"],
            "image_quality_tp1": post["image_quality"],
            "action_policy_raw": [theta_raw],
            "action_requested": [theta_norm],
            "action_executed": executed_action,
            "target": target,
            "execution": execution,
            "recovery": recovery,
            "obs_summary_t": pre["obs_summary"],
            "obs_summary_tp1": post["obs_summary"],
            "cvdm": cvdm_metrics,
            "reward_terms": reward_terms,
            "reward": float(reward),
            "motion": motion,
            "memory_update_gate": memory_update_detail,
        }
        self.logger.append(record)
        print(
            json.dumps(
                {
                    "step": self.step_count,
                    "reward": round(float(reward), 4),
                    "theta_raw": round(theta_raw, 3),
                    "theta_norm": round(theta_norm, 3),
                    "executed_action": [round(float(x), 3) for x in executed_action],
                    "distance_cm": round(float(motion["executed_distance_cm"]), 2),
                    "front_after_cm": motion["front_after_cm"],
                    "novelty_phi": round(float(cvdm_metrics.get("novelty_phi") or 0.0), 4),
                    "learning_progress": round(float(cvdm_metrics.get("learning_progress") or 0.0), 6),
                    "density_bank_size": cvdm_metrics.get("density_bank_size"),
                    "density_update_action": cvdm_metrics.get("density_update_action"),
                    "memory_update_ok": bool(memory_update_ok),
                    "frame": post["frame"]["relative_path"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        self.last_executed_action = executed_action.copy()
        self.current_state = post
        if self.args.sleep > 0:
            time.sleep(self.args.sleep)
        return post_obs.astype(np.float32), float(reward), False, False, {"record": to_jsonable(record)}

    def save_artifacts(self, model) -> dict[str, Any]:
        models_dir = self.logger.models_dir
        saved: dict[str, Any] = {}
        sac_path = models_dir / "sac_model.zip"
        model.save(sac_path)
        saved["sac_model"] = str(sac_path)
        try:
            replay_path = models_dir / "sac_replay.pkl"
            model.save_replay_buffer(replay_path)
            saved["sac_replay"] = str(replay_path)
        except Exception as exc:
            saved["sac_replay_save_error"] = repr(exc)
        full_path = models_dir / "cvdm_full.pt"
        saved["cvdm_full"] = save_cvdm_checkpoint(
            self.trainer,
            full_path,
            replay_state=self.replay.state_dict(),
            extra={
                "visual_encoder": self.visual_encoder.metadata() if hasattr(self.visual_encoder, "metadata") else {"kind": self.args.visual_encoder},
                "step_count": self.step_count,
                "run_name": self.args.run_name,
            },
        )
        saved["cvdm_modules"] = save_individual_model_params(self.trainer, models_dir)
        self.logger.write_manifest(
            {
                "run_name": self.args.run_name,
                "created_utc": self.start_wall_time,
                "finished_utc": utc_now_iso(),
                "out_dir": str(self.logger.out_dir),
                "steps_completed": self.step_count,
                "records_json": str(self.logger.json_path),
                "records_jsonl": str(self.logger.jsonl_path),
                "frames_dir": str(self.logger.frames_dir),
                "resume": self.resume_state,
                "models": saved,
                "cvdm_config": self.config.to_dict(),
                "args": vars(self.args),
            }
        )
        return saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CVDM + SAC online on the real rover.")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--run-name", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--resume-dir", default=None, help="Previous CVDM run directory to continue from.")
    p.add_argument("--resume-sac", default=None, help="Explicit Stable-Baselines SAC .zip to resume.")
    p.add_argument("--resume-cvdm", default=None, help="Explicit CVDM full checkpoint to resume.")
    p.add_argument("--no-resume-replay", action="store_true", help="Resume model weights/optimizers but start with empty SAC and CVDM replay buffers.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--visual-encoder", choices=["dino3", "hash"], default="dino3")
    p.add_argument("--dino-input-size", type=int, default=336)
    p.add_argument("--dino-threads", type=int, default=4)

    p.add_argument("--max-theta-deg", type=float, default=75.0)
    p.add_argument("--until-front-cm", type=float, default=40.0)
    p.add_argument("--until-front-max-seconds", type=float, default=1.5)
    p.add_argument("--cm-per-second", type=float, default=40.0)
    p.add_argument("--theta-deadzone", type=float, default=0.06)
    p.add_argument("--theta-power-gamma", type=float, default=2.0)
    p.add_argument("--turn-pwm", type=float, default=60.0)
    p.add_argument("--drive-pwm", type=float, default=65.0)
    p.add_argument("--front-stop-cm", type=float, default=35.0)
    p.add_argument("--front-clear-cm", type=float, default=45.0)
    p.add_argument("--no-echo-is-clear", action="store_true", help="Treat ultrasonic NO_ECHO as clear; default is unsafe/zero feature.")
    p.add_argument("--sleep", type=float, default=0.05)
    p.add_argument("--settle-seconds", type=float, default=0.35)

    p.add_argument("--learning-starts", type=int, default=25)
    p.add_argument("--sac-batch-size", type=int, default=32)
    p.add_argument("--sac-buffer-size", type=int, default=3000)
    p.add_argument("--sac-lr", type=float, default=3e-4)

    p.add_argument("--cvdm-batch-size", type=int, default=32)
    p.add_argument("--cvdm-gradient-steps", type=int, default=1)
    p.add_argument("--cvdm-replay-size", type=int, default=5000)
    p.add_argument("--cvdm-lr", type=float, default=2e-4)
    p.add_argument("--cvdm-rnd-lr", type=float, default=5e-5)
    p.add_argument("--cvdm-memory-size", type=int, default=2000)
    p.add_argument("--cvdm-memory-known-distance", type=float, default=0.35)
    p.add_argument("--cvdm-memory-norm-distance", type=float, default=1.25)
    p.add_argument("--cvdm-memory-min-assignment-margin", type=float, default=0.03)

    p.add_argument("--visual-min-motion-cm", type=float, default=10.0)
    p.add_argument("--visual-min-yaw-deg", type=float, default=5.0)
    p.add_argument("--visual-min-front-cm", type=float, default=25.0)
    p.add_argument("--visual-front-close-clear-cm", type=float, default=100.0)
    p.add_argument("--image-min-laplacian-var", type=float, default=25.0)
    p.add_argument("--image-min-mean", type=float, default=18.0)
    p.add_argument("--image-max-mean", type=float, default=238.0)
    p.add_argument("--image-min-std", type=float, default=8.0)
    p.add_argument("--image-max-dark-frac", type=float, default=0.55)
    p.add_argument("--image-max-bright-frac", type=float, default=0.45)

    p.add_argument("--safe-front-min-cm", type=float, default=35.0)
    p.add_argument("--safe-motion-min-cm", type=float, default=5.0)
    p.add_argument("--cvdm-novelty-weight", type=float, default=1.00)
    p.add_argument("--cvdm-learning-progress-weight", type=float, default=0.55)
    p.add_argument("--cvdm-distance-reward-weight", type=float, default=0.30)
    p.add_argument("--cvdm-safe-motion-bonus", type=float, default=0.10)
    p.add_argument("--cvdm-new-cluster-bonus", type=float, default=0.35)
    p.add_argument("--cvdm-contact-penalty", type=float, default=0.75)
    p.add_argument("--cvdm-zero-progress-penalty", type=float, default=0.22)
    p.add_argument("--cvdm-recovery-penalty", type=float, default=0.12)
    p.add_argument("--cvdm-near-obstacle-penalty", type=float, default=0.25)
    p.add_argument("--cvdm-obstructed-forward-penalty", type=float, default=0.45)
    p.add_argument("--cvdm-obstructed-forward-front-cm", type=float, default=40.0)
    p.add_argument("--cvdm-obstructed-forward-theta-deg", type=float, default=25.0)
    p.add_argument("--cvdm-clear-front-turn-penalty", type=float, default=0.05)
    p.add_argument("--cvdm-clear-front-turn-start-cm", type=float, default=45.0)
    p.add_argument("--cvdm-clear-front-turn-scale-cm", type=float, default=80.0)
    args = p.parse_args()
    if args.run_name is None:
        args.run_name = "cvdm_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir is None:
        args.out_dir = str(REPO_ROOT / "results" / args.run_name)
    return args


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    from stable_baselines3 import SAC

    env = RealCVDMSACEnv(args)
    model = None
    saved: dict[str, Any] = {}
    try:
        resume_sac = Path(args.resume_sac) if args.resume_sac else (Path(args.resume_dir) / "models/sac_model.zip" if args.resume_dir else None)
        if resume_sac is not None:
            if not resume_sac.exists():
                raise FileNotFoundError(f"SAC resume checkpoint not found: {resume_sac}")
            model = SAC.load(str(resume_sac), env=env, device="cpu")
            replay_path = Path(args.resume_dir) / "models/sac_replay.pkl" if args.resume_dir else None
            if replay_path is not None and replay_path.exists() and not args.no_resume_replay:
                model.load_replay_buffer(str(replay_path))
            print(json.dumps({"resume_sac": str(resume_sac), "resume_sac_replay": str(replay_path) if replay_path and replay_path.exists() and not args.no_resume_replay else None}, sort_keys=True), flush=True)
        else:
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=args.sac_lr,
                buffer_size=args.sac_buffer_size,
                learning_starts=args.learning_starts,
                batch_size=args.sac_batch_size,
                gamma=0.98,
                tau=0.02,
                train_freq=1,
                gradient_steps=1,
                policy_kwargs={"net_arch": [128, 128]},
                verbose=1,
                device="cpu",
                seed=args.seed,
            )
        model.learn(total_timesteps=args.steps, log_interval=1, progress_bar=False, reset_num_timesteps=(resume_sac is None))
        saved = env.save_artifacts(model)
        print(json.dumps({"saved": saved, "out_dir": str(env.logger.out_dir)}, sort_keys=True), flush=True)
    finally:
        try:
            if model is not None and not saved:
                saved = env.save_artifacts(model)
                print(json.dumps({"saved_on_finally": saved, "out_dir": str(env.logger.out_dir)}, sort_keys=True), flush=True)
        except Exception as exc:
            print(json.dumps({"artifact_save_error": repr(exc)}, sort_keys=True), flush=True)
        env.logger.finalize()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
