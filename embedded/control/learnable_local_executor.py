"""Tiny learnable wrapper around the deterministic local target executor."""

import os

import torch

from control.local_target_executor import LocalTargetExecutor, LocalTargetExecutorConfig


def _clamp(value, lo, hi):
    return max(lo, min(hi, float(value)))


def _norm_distance(value):
    if value is None:
        return -1.0
    return _clamp(value / 200.0, 0.0, 1.5)


class PrimitiveParamPolicy(torch.nn.Module):
    def __init__(self, input_dim=7, hidden=32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 4),
        )
        self.log_std = torch.nn.Parameter(torch.full((4,), -0.7))

    def forward(self, x):
        return self.net(x), self.log_std.clamp(-1.6, 0.4)


class LearnableLocalTargetExecutor:
    """Online policy-gradient learner for continuous primitive parameters.

    The policy cannot output motor ticks. It only chooses bounded parameters for
    `LocalTargetExecutor`: turn PWM, drive PWM, drive timing scale, and turn
    tolerance. Safety still gates all motion inside the deterministic executor.
    """

    def __init__(self, safety, checkpoint='data/web_learnable_executor.pt', lr=2e-3, status_callback=None):
        self.safety = safety
        self.checkpoint = checkpoint
        self.status_callback = status_callback
        self.policy = PrimitiveParamPolicy()
        if os.path.exists(checkpoint):
            self.policy.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.baseline = 0.0
        self.frozen = False

    def set_frozen(self, frozen):
        self.frozen = bool(frozen)
        return self.frozen

    def save(self):
        os.makedirs(os.path.dirname(self.checkpoint), exist_ok=True)
        torch.save(self.policy.state_dict(), self.checkpoint)

    def reset(self):
        self.policy = PrimitiveParamPolicy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.optimizer.param_groups[0]['lr'])
        self.baseline = 0.0
        if os.path.exists(self.checkpoint):
            os.remove(self.checkpoint)

    def make_state(self, x_cm, y_cm):
        distances = self.safety.read_distances()
        return torch.tensor([
            _clamp(x_cm / 200.0, -1.5, 1.5),
            _clamp(y_cm / 200.0, -1.5, 1.5),
            _clamp((x_cm * x_cm + y_cm * y_cm) ** 0.5 / 250.0, 0.0, 1.5),
            _norm_distance(distances['front']),
            _norm_distance(distances['left']),
            _norm_distance(distances['right']),
            1.0,
        ], dtype=torch.float32)

    def sample_params(self, state):
        mean, log_std = self.policy(state)
        if self.frozen:
            raw = mean
            log_prob = None
            entropy = None
        else:
            dist = torch.distributions.Normal(mean, log_std.exp())
            raw = dist.rsample()
            log_prob = dist.log_prob(raw).sum()
            entropy = dist.entropy().sum()
        z = torch.tanh(raw).detach().cpu().tolist()
        turn_pwm = 55.0 + (z[0] + 1.0) * 0.5 * 45.0
        drive_pwm = 70.0 + (z[1] + 1.0) * 0.5 * 30.0
        drive_time_scale = 0.65 + (z[2] + 1.0) * 0.5 * 1.0
        turn_tolerance = 3.0 + (z[3] + 1.0) * 0.5 * 10.0
        config = LocalTargetExecutorConfig(
            turn_pwm=turn_pwm,
            max_turn_pwm=100.0,
            drive_pwm=drive_pwm,
            turn_tolerance_deg=turn_tolerance,
            cm_per_second=40.0 / drive_time_scale,
        )
        params = {
            'turn_pwm': turn_pwm,
            'drive_pwm': drive_pwm,
            'drive_time_scale': drive_time_scale,
            'turn_tolerance_deg': turn_tolerance,
        }
        return config, params, log_prob, entropy

    @staticmethod
    def reward(report):
        reward = 1.0
        turn = report.get('turn') or {}
        drive = report.get('drive') or {}
        if not turn.get('ok'):
            reward -= 1.2
        else:
            target = max(1.0, abs(float(turn.get('target_deg', 0.0))))
            yaw = abs(float(turn.get('yaw_deg', 0.0)))
            reward -= min(1.0, abs(target - yaw) / 45.0)
        if not drive.get('ok'):
            reward -= 1.3
        else:
            reward += 0.7
        if 'front_safety_stop' in report.get('reason', ''):
            reward -= 1.0
        clipped = report.get('clipped_distance_cm', 0.0)
        requested = max(1.0, report.get('requested_distance_cm', 1.0))
        reward += 0.3 * min(1.0, clipped / requested)
        return reward

    def execute_local_target(self, x_cm, y_cm):
        state = self.make_state(x_cm, y_cm)
        config, params, log_prob, entropy = self.sample_params(state)
        if self.status_callback is not None:
            self.status_callback('learned_params ' + ','.join(f'{k}={v:.1f}' for k, v in params.items()))
        executor = LocalTargetExecutor(self.safety, config=config, status_callback=self.status_callback)
        report = executor.execute_local_target(x_cm, y_cm)
        report['learnable_params'] = params
        report['frozen'] = self.frozen
        report['reward'] = self.reward(report)
        if not self.frozen and log_prob is not None:
            self.baseline = 0.9 * self.baseline + 0.1 * report['reward']
            advantage = report['reward'] - self.baseline
            loss = -(log_prob * advantage) - 0.01 * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()
            self.save()
            report['loss'] = float(loss.detach().cpu())
        return report
