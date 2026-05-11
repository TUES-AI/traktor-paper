# Implementation Plan: Controllable Visual Dynamics Memory

## 0. Goal

Build a new rover learning stack where visual reward is not raw image novelty and not hand-patched visual clustering.

The central idea:

```text
Reward the rover for reaching new visual states that are predictable from its own actions.
```

This means:

```text
bad:  “this image is different, give reward”
good: “this action caused a new controllable visual transition, give reward”
```

The model should learn a controllable visual latent space from real rover transitions:

```text
current DINO visual state + range/action context + chosen action
    -> predicted next controllable visual state
```

Then SAC receives imagined candidate-action novelty scores without physically rotating the rover to scan directions.

## 1. What this replaces

This replaces three previous directions that caused problems:

### 1.1 Raw visual novelty

Raw DINO/PCVM novelty rewarded:

```text
blur
wall close-ups
collision/contact frames
recovery rotations
lighting artifacts
same place from a different angle
```

Raw visual novelty asks:

```text
is this frame far from previous frames?
```

But the real question should be:

```text
did the chosen action create useful new visual state?
```

### 1.2 Old path clusters

Old path clusters rewarded changes in a recurrent latent:

```text
camera + range + action + pose/context -> GRU -> latent z
```

Failure mode:

```text
GRU/context drift creates fake novelty
path bank explodes
robot loops but receives novelty reward
```

### 1.3 Physical scan every step

The idea:

```text
turn left -> image
turn right -> image
choose visual novelty direction
```

was conceptually useful but physically bad on the rover:

```text
too many body rotations
many zero-progress steps
many contact/stall events
bad action geometry near walls
```

The new model replaces physical scan with latent imagination:

```text
predict candidate futures in model space
do not rotate the robot just to inspect candidates
```

## 2. High-level architecture

At each real step:

```text
I_t                       camera frame
e_t = DINO(I_t)            frozen visual embedding, R^384
r_t                       range sensors, R^3
a_{t-1}                   last executed heading/action, R^1
phi_t = F(e_t, r_t, a_{t-1})  controllable latent, R^128

for candidate action a_i in {-75,-45,0,+45,+75}:
    phi_hat_i = T(phi_t, a_i)
    score_i = novelty(phi_hat_i)

SAC observes:
    phi_t
    range sensors
    last action
    candidate novelty scores
    candidate safety/clearance scores
    RND/surprise scalars

SAC chooses theta_norm
executor turns/drives with safety
I_{t+1}, execution feedback collected
phi_{t+1} computed
train dynamics/inverse/RND
reward from controllable novelty + learning progress + safety penalties
```

## 3. Module overview

### 3.1 Frozen visual encoder

Use the already-tested DINOv3 ONNX quantized encoder.

Initial choice:

```text
repo:    onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX
variant: model_quantized
input:   336x336 RGB
output:  384-dim pooler embedding
speed:   about 5 FPS on rover CPU at 336
```

Code should reuse or factor out logic from:

```text
VMM/pcvm_d3.py
tools/replay_dinov3_onnx_clusters.py
```

### 3.2 Controllable latent encoder

The encoder maps frozen visual features and small proprioceptive context to the controllable latent.

Input:

```text
e_t: normalized DINOv3 embedding, R^384
r_t: normalized range sensors, R^3
a_{t-1}: last executed theta_norm, R^1
```

Output:

```text
phi_t: controllable latent, R^128
```

First implementation should be a simple MLP, not GRU:

```text
reason: avoid hidden-state drift while validating the idea
```

Later add GRU only if offline tests show memory is needed.

### 3.3 Forward dynamics model

Predict the next controllable latent given current latent and action.

```text
T(phi_t, a_t) -> phi_hat_{t+1}
```

Input:

```text
phi_t: R^128
a_t:   R^1 theta_norm or executed theta_norm
```

Output:

```text
phi_hat_{t+1}: R^128
```

### 3.4 Inverse dynamics model

Predict which executed action caused a transition.

```text
I(phi_t, phi_{t+1}) -> a_hat_t
```

This helps force `phi` to encode action-relevant visual differences rather than arbitrary DINO variation.

### 3.5 Static/no-motion consistency

If the executor reports near-zero motion, `phi_t` and `phi_{t+1}` should be close.

```text
if executed_distance_cm < d_static:
    L_static = ||phi_t - phi_{t+1}||^2
```

This directly suppresses reward for:

```text
wall hits
stuck frames
zero-progress turns/drives
camera artifacts without motion
```

### 3.6 RND or density novelty in controllable latent

RND should run on `phi`, not raw images.

```text
rnd_error(phi) = ||predictor(phi) - frozen_target(phi)||^2
```

Alternative/additional density novelty:

```text
knn_novelty(phi) = min distance to prior phi memory
```

Start with both logged, but use only one in reward initially.

Recommended first reward source:

```text
learning_progress + small kNN novelty
```

Reason:

```text
pure RND can still reward representation drift if not carefully normalized
```

## 4. Data contract

Every transition stored in replay/logs should include:

```json
{
  "step": 12,
  "frame_t_path": "...",
  "frame_tp1_path": "...",
  "dino_t": "optional npy path or omitted if recomputable",
  "dino_tp1": "optional npy path or omitted if recomputable",
  "range_t": {"left": 123.0, "right": 87.0, "front": 210.0},
  "range_tp1": {"left": 110.0, "right": 95.0, "front": 54.0},
  "action_requested": [0.4],
  "action_executed": [0.36],
  "execution": {
    "turn": {"ok": true, "yaw_deg": 29.0},
    "drive": {"ok": true, "reason": "front_threshold_reached", "estimated_distance_cm": 74.2}
  },
  "safety": {
    "contact_or_stall": false,
    "recovery": false
  }
}
```

Do not depend on exact shape above; the implementation should parse existing logs too.

## 5. Model code skeleton

Create:

```text
VMM/controllable_visual_dynamics.py
```

Initial implementation sketch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


PHI_DIM = 128
DINO_DIM = 384
RANGE_DIM = 3
ACTION_DIM = 1


class ControllableEncoder(nn.Module):
    def __init__(self, phi_dim=PHI_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DINO_DIM + RANGE_DIM + ACTION_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, phi_dim),
        )

    def forward(self, dino, ranges, last_action):
        x = torch.cat([dino, ranges, last_action], dim=-1)
        phi = self.net(x)
        return F.normalize(phi, dim=-1)


class ForwardDynamics(nn.Module):
    def __init__(self, phi_dim=PHI_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(phi_dim + ACTION_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, phi_dim),
        )

    def forward(self, phi, action):
        pred = self.net(torch.cat([phi, action], dim=-1))
        return F.normalize(pred, dim=-1)


class InverseDynamics(nn.Module):
    def __init__(self, phi_dim=PHI_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(phi_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_DIM),
            nn.Tanh(),
        )

    def forward(self, phi_t, phi_tp1):
        return self.net(torch.cat([phi_t, phi_tp1], dim=-1))


class RND(nn.Module):
    def __init__(self, phi_dim=PHI_DIM, out_dim=128):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(phi_dim, 256), nn.ReLU(), nn.Linear(256, out_dim)
        )
        self.pred = nn.Sequential(
            nn.Linear(phi_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, out_dim)
        )
        for p in self.target.parameters():
            p.requires_grad = False

    def error(self, phi):
        with torch.no_grad():
            y = self.target(phi)
        yhat = self.pred(phi)
        return F.mse_loss(yhat, y, reduction="none").mean(dim=-1)
```

Wrapper module:

```python
class ControllableVisualDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ControllableEncoder()
        self.forward_model = ForwardDynamics()
        self.inverse_model = InverseDynamics()
        self.rnd = RND()

    def encode(self, dino, ranges, last_action):
        return self.encoder(dino, ranges, last_action)

    def predict_candidates(self, phi, candidate_actions):
        # phi: [B, D]
        # candidate_actions: [K, 1]
        B = phi.shape[0]
        K = candidate_actions.shape[0]
        phi_rep = phi[:, None, :].expand(B, K, -1).reshape(B * K, -1)
        a_rep = candidate_actions[None, :, :].expand(B, K, -1).reshape(B * K, -1)
        pred = self.forward_model(phi_rep, a_rep)
        return pred.reshape(B, K, -1)
```

## 6. Loss functions

Transition batch fields:

```text
dino_t, range_t, last_action_t
action_t
dino_tp1, range_tp1, last_action_tp1
executed_distance_cm
contact/recovery flags
```

Loss function:

```python
def compute_losses(model, batch):
    phi_t = model.encode(batch.dino_t, batch.range_t, batch.last_action_t)
    phi_tp1 = model.encode(batch.dino_tp1, batch.range_tp1, batch.action_t)

    pred_tp1 = model.forward_model(phi_t, batch.action_t)
    action_hat = model.inverse_model(phi_t.detach(), phi_tp1.detach())

    forward_loss = F.mse_loss(pred_tp1, phi_tp1.detach())
    inverse_loss = F.mse_loss(action_hat, batch.action_executed)

    static_mask = (batch.executed_distance_cm < 3.0).float().view(-1, 1)
    static_loss = (static_mask * (phi_t - phi_tp1).pow(2)).mean()

    rnd_error = model.rnd.error(phi_tp1.detach())
    rnd_loss = rnd_error.mean()

    loss = (
        forward_loss
        + 0.2 * inverse_loss
        + 0.5 * static_loss
        + 0.1 * rnd_loss
    )

    return loss, {
        "forward_loss": forward_loss.item(),
        "inverse_loss": inverse_loss.item(),
        "static_loss": static_loss.item(),
        "rnd_loss": rnd_loss.item(),
    }
```

Important: the RND predictor optimizer may be separate from the dynamics optimizer, but for first version keep it simple.

## 7. Learning progress reward

Need a function to estimate whether the transition was learnable.

Simple online version:

```python
@torch.no_grad()
def transition_error(model, batch):
    phi_t = model.encode(batch.dino_t, batch.range_t, batch.last_action_t)
    phi_tp1 = model.encode(batch.dino_tp1, batch.range_tp1, batch.action_t)
    pred = model.forward_model(phi_t, batch.action_t)
    return (pred - phi_tp1).pow(2).mean(dim=-1)


def train_step_with_learning_progress(model, opt, batch):
    err_before = transition_error(model, batch)
    loss, metrics = compute_losses(model, batch)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    err_after = transition_error(model, batch)
    lp = torch.clamp(err_before - err_after, min=0.0)
    return lp.detach(), metrics
```

Potential issue:

```text
single SGD step noise can make learning_progress noisy
```

Better later:

```text
EMA of prediction error per transition class / cluster
or compare error before/after several replay updates
```

For first experiment, log it but do not rely on it as the only reward.

## 8. Candidate action imagination

Candidate action set:

```python
CANDIDATE_ACTIONS = torch.tensor([[-1.0], [-0.6], [0.0], [0.6], [1.0]], dtype=torch.float32)
```

Mapping with max theta 75 degrees:

```text
-1.0 -> -75°
-0.6 -> -45°
 0.0 ->   0°
 0.6 -> +45°
 1.0 -> +75°
```

Scoring imagined futures:

```python
@torch.no_grad()
def candidate_scores(model, phi, candidate_actions, density_memory=None):
    pred_phi = model.predict_candidates(phi, candidate_actions)  # [B, K, D]
    B, K, D = pred_phi.shape
    flat = pred_phi.reshape(B * K, D)

    rnd = model.rnd.error(flat).reshape(B, K)

    if density_memory is not None:
        density = density_memory.knn_distance(flat).reshape(B, K)
    else:
        density = rnd

    scores = normalize_candidate_scores(0.5 * rnd + 0.5 * density)
    return scores, pred_phi
```

Normalization:

```python
def normalize_candidate_scores(scores):
    # scores: [B, K]
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True).clamp_min(1e-6)
    z = (scores - mean) / std
    return torch.sigmoid(z)
```

Do not reward candidate scores. They are only SAC inputs.

## 9. SAC observation design

Initial SAC observation vector:

```text
phi_t                         128
range_norm                    3
last_action                   1
candidate_novelty_scores      5
candidate_clearance_scores    5
rnd_current                   1
transition_surprise_current   1
--------------------------------
total                         144
```

No raw pixels.

No IMU yaw-rate if testing vision-dominant setup.

IMU stays in the executor and safety layer.

Candidate clearance can be cheap:

```python
def candidate_clearance_scores(distances):
    # For theta-front action candidates, approximate:
    front = norm_distance(distances.get("front"))
    left = norm_distance(distances.get("left"))
    right = norm_distance(distances.get("right"))
    return np.array([
        right,                 # -75
        0.5 * (right + front), # -45
        front,                 # 0
        0.5 * (left + front),  # +45
        left,                  # +75
    ], dtype=np.float32)
```

## 10. Reward design

Start simple.

```python
def reward_from_transition(
    novelty,
    learning_progress,
    executed_distance_cm,
    front_after_cm,
    contact_or_stall,
    recovery,
    zero_progress,
):
    safe_motion = (
        executed_distance_cm > 5.0
        and not contact_or_stall
        and not recovery
        and front_after_cm is not None
        and front_after_cm > 35.0
    )

    r = 0.0
    if safe_motion:
        r += 1.0 * novelty
        r += 0.5 * learning_progress
        r += 0.2 * min(1.0, executed_distance_cm / 120.0)

    if zero_progress:
        r -= 0.3
    if contact_or_stall:
        r -= 1.0
    if recovery:
        r -= 0.6
    if front_after_cm is not None and front_after_cm < 35.0:
        r -= 0.4 * ((35.0 - front_after_cm) / 35.0) ** 2

    return float(r)
```

This still uses physical gates, but fewer visual-specific patches.

Avoid adding blur/wall/lighting filters in version 1. The point of this model is to see whether controllable representation already suppresses those.

## 11. Implementation phases

### Phase 1: offline dataset builder

Create:

```text
tools/build_cvd_dataset.py
```

Inputs:

```text
--log results/<run>.jsonl
--frame-dir results/<run>_frames
--out results/<run>_cvd_dataset.npz
```

Outputs:

```text
dino_t:               [N, 384]
dino_tp1:             [N, 384]
range_t:              [N, 3]
range_tp1:            [N, 3]
last_action_t:        [N, 1]
action_executed:      [N, 1]
executed_distance_cm: [N]
front_after_cm:       [N]
contact_or_stall:     [N]
recovery:             [N]
frame_t_path:         list[str]
frame_tp1_path:       list[str]
```

Dataset pairing:

```text
transition i = row i -> row i+1
```

For older logs where only post-action frames exist, use consecutive frames as transitions.

### Phase 2: offline model trainer

Create:

```text
tools/train_controllable_visual_dynamics.py
```

Inputs:

```text
--dataset results/<run>_cvd_dataset.npz
--out results/cvd_model.pt
--epochs 100
```

Metrics to print:

```text
forward_loss
inverse_loss
static_loss
RND error mean/std
candidate score entropy
```

### Phase 3: offline novelty audit

Create:

```text
tools/audit_controllable_visual_novelty.py
```

Outputs:

```text
top_raw_dino_novelty.png
top_phi_novelty.png
top_learning_progress.png
reward_timeline.png
candidate_scores_timeline.png
audit_summary.json
```

Critical question:

```text
Do top phi novelty / learning-progress frames look like rooms, doors, corridors,
or do they still look like walls, blur, recovery, contact?
```

### Phase 4: online wrapper without SAC

Before SAC, run a passive online diagnostic:

```text
normal theta-front policy from old SAC checkpoint
CVD model observes transitions
logs candidate predictions and rewards
does not control policy
```

Create:

```text
embedded/scripts/run_real_cvd_diagnostic.py
```

Purpose:

```text
verify rover CPU speed
verify online update stability
verify candidate scores are sensible
```

### Phase 5: online SAC training

Create:

```text
embedded/scripts/train_real_cvd_sac.py
```

Start from scratch first because observation shape is new.

Run:

```bash
/home/yasen/.venv/bin/python embedded/scripts/train_real_cvd_sac.py \
  --steps 100 \
  --save-path results/cvd_sac_100.zip \
  --log-path results/cvd_sac_100.jsonl \
  --frame-dir results/cvd_sac_100_frames
```

## 12. Offline testing plan

Use the saved runs we already have:

```text
results/pcvm_m_from_latest_sac_100_20260510.jsonl
results/pcvm_m_from_latest_sac_100_20260510_frames/

results/pcvm_d3_raw_from_pre_dino_sac_100_20260511.jsonl
results/pcvm_d3_raw_from_pre_dino_sac_100_20260511_frames/

results/dino_scan_sac_50_20260511_max6.jsonl
results/dino_scan_sac_50_20260511_max6_frames/
```

Tests:

```text
1. dataset builds without missing frames
2. DINO embeddings cache correctly
3. model trains without collapse
4. inverse model predicts left/center/right better than random
5. static loss makes zero-motion transitions close in phi-space
6. top novelty frames are manually inspectable
7. candidate scores vary meaningfully across a trajectory
```

Collapse tests:

```python
phi = model.encode(...)
print(phi.std(dim=0).mean())
print(torch.pdist(phi[:200]).mean())
```

Expected:

```text
not near zero
not exploding
```

Inverse sanity:

```python
pred = inverse(phi_t, phi_tp1)
corr = np.corrcoef(pred[:,0], action_executed[:,0])[0,1]
```

Expected:

```text
positive correlation with executed heading
```

Static sanity:

```text
mean ||phi_t - phi_tp1|| for zero-motion transitions
should be lower than for real-motion transitions
```

## 13. Online testing plan

### 13.1 CVD diagnostic run

Run normal old SAC/executor, but CVD only logs.

Success criteria:

```text
no runtime crashes
DINO + CVD inference under 0.5 sec per decision
candidate scores logged every step
top predicted candidate sometimes corresponds to visually plausible direction
```

### 13.2 First CVD-SAC run

Run 50 steps only.

Success criteria:

```text
not necessarily good exploration
no repeated zero-motion exploit
reward low/negative during contact/recovery
candidate novelty scores not constant
visual/dynamics model state saved
```

### 13.3 100-step run

Only after 50-step sanity.

Compare against:

```text
old theta-front PCVM-M run
PCVM-D3 run with visual clusters
no-visual baseline if available
```

## 14. Logging contract for online CVD

Each step should log:

```json
{
  "step": 1,
  "obs_summary": {
    "candidate_scores": [0.1, 0.4, 0.2, 0.7, 0.5],
    "candidate_clearance": [0.2, 0.4, 0.8, 0.7, 0.5],
    "rnd_current": 0.13,
    "transition_surprise": 0.08
  },
  "action": [0.6],
  "execution": {},
  "cvd": {
    "novelty_phi": 0.42,
    "learning_progress": 0.09,
    "forward_error_before": 0.17,
    "forward_error_after": 0.08,
    "inverse_error": 0.12,
    "static_transition": false
  },
  "reward_terms": {
    "novelty_reward": 0.42,
    "learning_progress_reward": 0.045,
    "distance_reward": 0.08,
    "contact_penalty": 0.0,
    "zero_progress_penalty": 0.0,
    "recovery_penalty": 0.0
  },
  "reward": 0.545
}
```

## 15. Model persistence

Need sidecar save just like PCVM.

Artifacts:

```text
SAC: results/name.zip
CVD: results/name_cvd.pt
replay: results/name_replay.pkl
frames: results/name_frames/
log: results/name.jsonl
```

CVD checkpoint includes:

```text
encoder state_dict
forward_model state_dict
inverse_model state_dict
RND target/predictor state_dict
optimizers
density memory if used
normalization stats
```

## 16. Candidate safety/clearance

Do not let imagined visual novelty blindly choose walls.

Candidate score for SAC should include both:

```text
predicted novelty
cheap clearance estimate
```

But do not multiply them into one irreversible scalar only. Give SAC both.

Observation includes:

```text
candidate_novelty[5]
candidate_clearance[5]
```

This lets SAC learn tradeoffs.

## 17. Important implementation warnings

### 17.1 Do not update memory from imagined candidates

Candidate predictions are counterfactual.

Bad:

```text
predict candidate phi -> add to memory
```

Good:

```text
predict candidate phi -> score only
actual post-action phi -> update memory/RND/training
```

### 17.2 Do not reward prediction error alone

Prediction error rewards chaos.

Prefer:

```text
learning progress
or novelty gated by safe executed motion
```

### 17.3 Do not make the encoder too recurrent too early

A GRU can reintroduce hidden-state drift.

Start with MLP encoder.

Add recurrent context only after offline audits show it is needed.

### 17.4 Keep IMU out of SAC for the vision-dominant test

Use IMU for:

```text
turn execution
safety
executed feedback labels
```

Do not include raw yaw/pose as SAC observation in the first vision-dominant CVD test.

## 18. Paper experiment plan

Ablations:

```text
A. executor + safety + distance only
B. raw DINO novelty
C. DINO visual clusters
D. grounded DINO clusters
E. controllable visual dynamics novelty without candidate imagination
F. controllable visual dynamics + imagined candidate scores + SAC
```

Metrics:

```text
human trajectory label
contacts/recoveries
zero-progress fraction
visual artifact reward rate
top-novelty frame purity
coverage/travel as secondary metrics
candidate-score interpretability
```

Figures:

```text
1. architecture diagram
2. raw novelty vs controllable novelty top-frame sheet
3. candidate imagined novelty over time
4. reward decomposition
5. trajectory/contact sheet
```

Main expected result:

```text
raw visual novelty rewards artifacts;
controllable visual dynamics shifts reward toward action-caused visual progress.
```

## 19. Immediate next concrete tasks

1. Implement `VMM/controllable_visual_dynamics.py`.
2. Implement `tools/build_cvd_dataset.py` for existing logs.
3. Train offline CVD on the latest good 100-step run.
4. Generate novelty audit sheets.
5. Decide if top novelty is sane before any rover run.
6. Implement passive online diagnostic.
7. Only then implement online SAC environment.

Do not start with online SAC. The model must first prove offline that controllable novelty ranks good frames above wall/contact/recovery artifacts.
