# MapExRL: Predicted Environment Context and Reinforcement Learning for Exploration

Source: arXiv `2503.01548`

## Problem and core idea

MapExRL studies efficient indoor exploration under a distance budget. It argues that primitive-motion RL is too myopic for large environments, and that a learned policy should select frontiers/subgoals using predicted global context.

## Method details

- Conducts a human user study to identify good exploration strategies.
- Uses partial occupancy maps plus an ensemble of global map prediction models.
- SAC policy scores a fixed set of top-N frontiers, rather than outputting primitive movement.
- Observation includes encoded predicted map, frontier coordinates, utility scores, prediction uncertainty, A* distance, and remaining budget.
- Reward is sparse and terminal: predicted-map IoU plus remaining budget.

## Key results

- Reports up to `18.8%` improvement over the strongest SOTA baseline in some settings.
- Overall combined evaluation improves about `4.8%` over MapEx, `30%` over random frontier selection, and `55%` over nearest-frontier exploration.
- Motion-primitive PPO baseline often behaves like a random walk in large complex maps; training mostly reduces collisions rather than learning efficient exploration.
- The method is strongest on large/topologically complex maps where long-horizon choice matters.

## Relevance to this project

High as a negative/abstraction lesson, low as a direct method. It strongly supports our decision not to learn raw motor commands: use a high-level local-target/frontier/action abstraction and let classical/safety logic handle execution.

## Concrete experiments to run next

- Keep SAC action space as local targets/arcs, not motor PWM.
- Score a small candidate set of local moves using novelty, distance safety, path-revisit penalty, and predicted open-space value.
- Add a distance/time budget term to short rover episodes.
- Include a motion-primitive baseline only as a negative ablation if we can do it safely in sim, not on the real rover.

## Risks / open questions

- Uses explicit occupancy maps, A*, 2D LiDAR, known pose, and map-prediction IoU; this violates our final no-map story.
- Evaluation is floorplan prediction, not real onboard coverage.
- The useful transfer is action abstraction and budget-aware scoring, not map prediction.
