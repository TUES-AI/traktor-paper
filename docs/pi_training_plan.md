# Pi Real-World SAC + RND Training Plan

## Phase 0 — Hardware prep

1. Fix camera + 3 ultrasonic sensors to rover, verify all read at boot
2. Set Pi to performance CPU governor so inference time is consistent
3. Mount a UPS hat or large powerbank — power cuts mid-training corrupt weights
4. SSH into Pi headlessly, run everything in `tmux` so disconnects don't kill training
5. Clear a ~3×3m room, tape the boundaries, place 2–3 fixed obstacles that won't move session to session

---

## Phase 1 — Install & verify stack

6. Install PyTorch (ARM wheel), torchvision, `stable-baselines3`, `picamera2`, RPi GPIO libs into the venv at `/home/yasen/traktor-venv`
7. Verify MobileNetV3-Small inference time on Pi — must stay under 150ms per frame
8. Verify ultrasonic read latency — all 3 sensors must complete within 50ms
9. Confirm total step budget: inference + sensors + motor command must fit in ~400ms, leaving 100ms for gradient update

---

## Phase 2 — Build the data pipeline (one script)

10. Camera → MobileNetV3-Small (pretrained, frozen) → 128-dim embedding
11. Normalise embedding online with running mean/std (Welford's algorithm)
12. RND: two small MLPs (256→256), fixed target + trained predictor, both taking the 128-dim embedding
13. RND reward = MSE(predictor(e), target(e)), clipped to `[0, rnd_max]` where `rnd_max` is set from the first 200 steps
14. State vector: `[embedding(128) | rnd_reward_normalised(1) | last_vl(1) | last_vr(1) | d_left(1) | d_right(1) | d_front(1)]` = 134 floats
15. Action: continuous `(speed, turn)` → mapped to `(vl, vr)` before sending to motors
16. Safety clamp wraps motor output — log the **clamped** `(vl, vr)` into replay buffer, not the raw SAC output

---

## Phase 3 — Warm-start with reactive agent

17. Run the reactive agent (ported to Pi) for 1,000 steps
18. Every step: record `(state, clamped_action, rnd_reward, next_state)` into the replay buffer
19. Do **no gradient updates** during this phase — just collect safe transitions
20. After 1,000 steps, freeze this buffer as the starting point; save it to disk

---

## Phase 4 — SAC training loop

21. Load pre-filled buffer, initialise SAC actor + 2 critics (small MLPs: 256→256→2)
22. Each step:
    - Read sensors + capture frame → build state
    - SAC actor forward pass → `(speed, turn)`
    - Apply safety clamp → `(vl, vr)` sent to motors
    - Rover moves for 0.5s
    - Read next sensors + frame → build next state
    - Compute RND reward
    - Push `(state, clamped_action, rnd_reward, next_state, done)` to replay buffer
    - Every 5 steps: one gradient update on SAC + one gradient update on RND predictor
23. Checkpoint every 200 steps: save actor, critics, RND predictor, replay buffer, running stats
24. If collision detected by clamp: log it, do **not** end episode — just continue (real world has no reset)

---

## Phase 5 — RND stabilisation safeguards

25. For first 500 steps after SAC turns on: clip RND reward to 50% of its running max — prevents Q-function inflation during cold start
26. After 500 steps: release clip, let full RND reward flow
27. Monitor predictor loss — if it stops decreasing the rover has stopped visiting new places (coverage saturated or stuck loop)

---

## Phase 6 — Evaluation checkpoints

28. Every 2,000 steps: pause training, run reactive agent for 100 steps, measure how often SAC-trained policy gets clamped vs reactive baseline — fewer clamp triggers = policy is learning safe navigation
29. Track RND predictor loss over time — should decrease as the room gets covered
30. When predictor loss plateaus for 500+ steps: room is covered, move obstacles or expand the space

---

## Practical notes

- **Never** run without the safety clamp active, even after 10,000 steps
- Run training in sessions of 1–2 hours max, resume from checkpoint — Pi thermals under sustained load
- The replay buffer will grow large; cap it at 50,000 transitions and use uniform sampling
- If the rover gets stuck in a physical corner (cables, etc.) and the clamp fires for 20+ consecutive steps: manually reposition and inject a "done" flag into the buffer so SAC doesn't learn that position is normal

---

## Ablations

### Selected ablations (1–4)

1. **SAC+RND vs reactive agent baseline** — primary comparison. The reactive agent loops predictably; SAC+RND should find strictly more novel places over the same number of steps
2. **RND vs no intrinsic reward** — SAC with only collision penalty vs SAC+RND. Isolates whether novelty-seeking drives exploration or the policy discovers it anyway
3. **Clamped action in buffer vs raw SAC action** — directly tests the corruption hypothesis. Compare Q-function stability and novel places found across identical step counts
4. **Camera + sensors vs sensors only** — sensors-only policy cannot compute RND at all, making this the clearest possible control condition


### Primary evaluation metric — novel places found

Coverage % is not usable in the real world without ground truth position (GPS/odometry). Instead:

- Run a fixed-length evaluation episode (frozen weights, no training, same room layout)
- Collect all 128-dim embeddings from that episode
- Cluster with k-means (k=50) and count distinct clusters visited — this is the reported number
- Secondary metric: count of steps where RND reward exceeds a threshold set from the warm-start baseline

**Why this is stronger than coverage:**
- Computed entirely from the onboard camera — no external instrumentation needed
- Directly measures what RND optimises, so metric and objective are aligned
- Reproducible across sessions without repositioning the rover identically

### What each ablation shows under this metric

| Ablation | Expected result |
|---|---|
| SAC+RND vs reactive | Reactive loops → low cluster count; SAC+RND finds more |
| RND vs no intrinsic | No intrinsic reward → policy has no reason to seek novelty → cluster count drops |
| Clamped vs raw action | Both may find similar clusters but raw-action shows higher clamp rate and unstable learning curve |
| Camera+sensors vs sensors only | Sensors-only cannot run RND — becomes the zero-novelty control |

### The one result that would make the paper

Ablation 3 — clamped action in buffer. Nobody tests this explicitly in real-world RL with safety layers. A learning curve comparison (novel places found over steps, two conditions) is a concrete, replicable finding rather than an engineering footnote.

---

## RLxF @ ICML 2026 — Submission Plan

**Deadline: May 13, 2026 — 11 days away**
**Target format: short paper, 2–4 pages**
**Workshop:** [RLxF — Reinforcement Learning from World Feedback](https://sites.google.com/view/rlxf-icml2026)

### Why this fits

The workshop is explicitly about using real-world physical signals as the learning signal instead of human labels. This system uses sensor readings, collision signals, and RND prediction error on real camera frames — all "world feedback" in their framing. Physical Pi deployment gives it hardware credibility most workshop submissions lack.

### Paper structure

**Body (~3 pages)**
- System description — architecture, pipeline, the clamped-action-in-buffer argument as a design principle (framed as a contribution, not yet fully empirically tested)
- Novel places found as a proposed metric — argument for why it's better than coverage for real-world eval without instrumentation
- Training procedure on physical hardware

**Results (~1 page)**
- Learning curve: novel places found (cluster count) over training steps — single run
- RND predictor loss decreasing over time — shows the room is being explored
- These two curves together tell the exploration story without requiring a comparison baseline

### Low-cost ablations from a single training run

All three require no extra runs — just proper logging during the one training session:

1. **RND normalisation** — log both raw and normalised reward during training, show stability difference in one plot
2. **Warm-start effect** — mark the 1,000-step boundary on the learning curve, show the transition where SAC takes over from the reactive agent
3. **Clamp trigger rate over time** — log how often the safety clamp fires as training progresses; decreasing rate = policy is learning to avoid walls on its own

### Narrative

> *"We propose a real-world RL training pipeline for physical rovers using world feedback alone, demonstrate it on hardware, and introduce novel places found as an evaluation metric deployable without external instrumentation."*

### Timeline

| Task | Days remaining |
|---|---|
| Start training run on Pi | Days 1–2 |
| Collect enough steps for learning curves | Days 2–6 |
| Write system + metric sections | Days 3–5 |
| Generate plots from logs | Days 6–8 |
| Polish + submit | Days 9–11 |
