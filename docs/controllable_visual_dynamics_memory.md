# Controllable Visual Dynamics Memory

## One-sentence idea

Do not reward the rover because an image looks new. Reward it when its chosen action causes a new visual state that the robot can learn to predict and revisit.

In short:

```text
raw visual novelty:          “is this frame different?”
controllable visual novelty: “did my action lead to a new controllable visual state?”
```

This changes the role of vision from a noisy novelty detector into part of a learned world model.

## Why raw visual novelty failed

Raw DINO/PCVM novelty asks whether the current frame is far from previous frames. That sounds useful, but on a real rover the most visually novel frames are often bad frames:

```text
blur
wall close-ups
collision/contact frames
recovery rotations
lighting artifacts
weird camera exposure
same place from a slightly different angle
```

So the robot can receive reward for states that are visually different but physically useless.

That is the core mistake:

```text
different image != useful exploration
```

## The new target

The new target is not visual difference. The new target is action-caused visual progress.

The model learns:

```text
current visual state + action -> next visual state
```

Then novelty is measured in the learned controllable state space, not directly in pixel/DINO space.

So the reward becomes:

```text
reward = “I reached a new visual state that my action can explain/control”
```

not:

```text
reward = “the camera frame looks different”
```

## Model wiring

At time `t`:

```text
camera frame I_t
    -> frozen DINOv3 encoder
    -> visual embedding e_t in R^384

e_t + range sensors + last executed action
    -> learned controllable encoder
    -> phi_t in R^128
```

`phi_t` is the robot's learned controllable visual state.

It should keep visual information that is useful for predicting action-conditioned future observations, and ignore visual changes that are not controllable.

The dynamics model predicts future latent states:

```text
phi_hat_{t+1} = T(phi_t, action_t)
```

The inverse model predicts what action caused a transition:

```text
action_hat_t = I(phi_t, phi_{t+1})
```

Random Network Distillation (RND) or density novelty runs on `phi`, not raw pixels:

```text
novelty(phi_t) = RND error or kNN/prototype distance in controllable latent space
```

## How this handles blur, walls, and artifacts

The important difference is that the representation is trained around controllability.

A blur frame may be visually different, but it is not a stable consequence of a chosen action.

```text
action + state -> blur
```

is not a reliable transition. The model should not build its useful state space around it.

A lighting flicker is visually different, but not caused by the robot's action.

A wall close-up after collision is visually different, but it is paired with bad executor feedback:

```text
zero progress
contact/stall
recovery
front range too small
```

So it is not a useful controllable transition.

A doorway after turning and driving is different in the right way:

```text
corridor view + turn/drive action -> doorway/room view
```

That is action-caused, repeatable, and useful.

This is why the model should reward controllable visual change instead of raw visual change.

## Where the reward comes from

After the rover executes an action, we have:

```text
phi_t          current controllable latent
action_t       chosen relative heading
phi_{t+1}      reached controllable latent
executor info  distance, contact, recovery, front range
```

There are two main reward sources.

### 1. Novelty in controllable latent space

Ask whether `phi_{t+1}` is rare or far from previous controllable states:

```text
r_novel = kNN_distance(phi_{t+1}, memory)
```

or with RND:

```text
r_rnd = || predictor(phi_{t+1}) - frozen_random_target(phi_{t+1}) ||^2
```

This gives intrinsic reward for reaching new controllable states.

### 2. Learning progress

Prediction error alone can reward chaos. Learning progress is better.

Before training the transition model on the new transition:

```text
error_before = || T_old(phi_t, action_t) - phi_{t+1} ||^2
```

After training:

```text
error_after = || T_new(phi_t, action_t) - phi_{t+1} ||^2
```

Reward the improvement:

```text
r_learning_progress = max(0, error_before - error_after)
```

Interpretation:

```text
high error and learnable = useful new experience
high error and not learnable = noise/artifact
```

## Final reward sketch

The reward can be:

```text
r_t =
    w_novel * novelty(phi_{t+1})
  + w_lp    * learning_progress(phi_t, action_t, phi_{t+1})
  + w_dist  * small safe executed-distance reward
  - w_contact  * contact_or_stall
  - w_zero     * zero_progress
  - w_recovery * recovery_event
  - w_near     * near_obstacle_penalty
```

The safety/executor terms are still needed, but they are not the main exploration signal. They say whether the transition was physically valid.

## How the world model helps SAC know where to turn

This is the key part.

At decision time, the robot does not physically scan left/right. Instead, the transition model imagines possible futures.

Current latent:

```text
phi_t
```

Candidate actions:

```text
-75°, -45°, 0°, +45°, +75°
```

For each candidate action:

```text
phi_hat_next(action) = T(phi_t, action)
```

Then score each imagined future:

```text
score(action) = novelty(phi_hat_next(action))
```

SAC receives these candidate scores as part of its observation:

```text
current latent phi_t
range sensors
last action
predicted novelty if turning -75
predicted novelty if turning -45
predicted novelty if going forward
predicted novelty if turning +45
predicted novelty if turning +75
```

So SAC gets a mental lookahead signal:

```text
which heading is predicted to lead to a new controllable visual state?
```

This replaces the bad physical scan idea.

Old active scan:

```text
rotate robot left/right -> take images -> choose direction
```

New model:

```text
imagine left/right futures in latent space -> choose direction
```

## Why this is better than clustered visual novelty

Clustered visual novelty says:

```text
is this view a new stable visual place?
```

That still needs many hand gates:

```text
reject blur
reject wall close-up
reject recovery
reject zero-motion
reject bad lighting
confirm clusters over time
separate artifact clusters
```

Controllable visual dynamics changes the question:

```text
is this new visual state a consequence of my action?
```

Many artifacts become less important because they are not action-consistent, controllable transitions.

So the system needs fewer visual-specific patches.

It still needs basic physical penalties:

```text
contact
zero progress
recovery
near obstacle
```

but it no longer needs a special rule for every kind of bad image novelty.

## What this replaces

This should replace the old meaning of PCVM/path clusters.

Old PCVM path novelty:

```text
GRU hidden changes -> cluster says new -> reward
```

Failure:

```text
hidden-state drift can be rewarded even during loops
```

New controllable visual dynamics:

```text
DINO visual state + action -> predicted controllable next state
actual transition updates model
reward is novelty/learning progress in that controllable state space
```

The GRU or recurrent state can still exist, but it should not be rewarded just for drifting.

## What DINO does

DINO is not the reward by itself.

DINO provides a strong visual feature basis:

```text
image -> e_t
```

The learned dynamics model decides which parts of `e_t` matter for action-conditioned future prediction.

This is important:

```text
DINO = perceptual backbone
world model = controllability filter
RND/density = curiosity in controllable state space
SAC = action selection
executor = physical grounding
```

## Paper framing

The paper claim becomes:

```text
Ungrounded visual novelty can mislead real robots.
We instead learn a controllable visual dynamics space from onboard world feedback.
Intrinsic reward is computed in this controllable latent space, so novelty corresponds to action-caused expansion of the robot's world model rather than arbitrary image difference.
```

This fits reinforcement learning from world feedback because the world feedback is not just reward. It teaches the model which visual changes are real, controllable, safe, and useful.

## Minimal implementation path

First offline:

```text
use saved rover logs
compute DINO embeddings
train f, T, inverse model on transitions
compare raw DINO novelty vs controllable-latent novelty
inspect top novelty frames
```

Then online:

```text
normal theta-front executor
no physical scan
SAC sees candidate novelty predicted by the transition model
reward uses controllable novelty + learning progress + safety penalties
```

The first question to answer is simple:

```text
Do top novelty states become rooms, doorways, and new corridor views instead of walls, blur, and recovery frames?
```

If yes, this is a much stronger direction than patching visual clusters.
