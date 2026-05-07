# CREStE: Scalable Mapless Navigation with Internet Scale Priors and Counterfactual Guidance

Source: arXiv `2503.03921`, RSS 2025, repo: `https://github.com/ut-amrl/creste_public`

## Problem and core idea

CREStE tackles urban mapless navigation where geometry alone is insufficient and the relevant semantic/social factors are open-ended. The core recipe is to distill visual foundation model priors into a structured BEV representation, then learn a reward/cost map from expert and counterfactual negative trajectories.

## Method details

- Perceptual encoder takes RGB plus sparse depth and predicts completed depth plus structured BEV features.
- Distills DINOv2 semantic features and uses SAM2-derived instance labels for static/dynamic panoptic BEV structure.
- Reward network is a small multi-scale fully convolutional network over BEV features.
- Counterfactual IRL penalizes bad alternate paths under the same start/goal/observation context.
- Runtime planner enumerates `31` constant-curvature arcs, samples `30` points per arc, discounts by `0.95`, and chooses the lowest-cost arc.

## Key results

- Uses only about `3` hours of expert demonstrations.
- Annotates negative counterfactuals for only `3%` of training data.
- Counterfactuals reduce interventions by about `70%` in seen environments and `69%` in unseen environments.
- Long-horizon run: about `1919 m`, `99.45%` subgoals reached, `1` intervention.
- Removing structured BEV causes `28%` more interventions; removing both structure and counterfactuals causes `41%` more interventions.

## Relevance to this project

Very high conceptually, but not directly portable. The transferable parts are: structured representation beats raw latent maps, targeted negative examples are extremely valuable, and the final controller should remain an interpretable local-trajectory selector rather than a raw policy.

## Concrete experiments to run next

- Generate synthetic “bad alternatives” from WASD segments: same start/end intent, but paths that hug obstacles, revisit, or stall.
- Label/derive negatives offline from recovery logs and near-obstacle events.
- Train a small local-score head to rank candidate local targets/arcs using positive WASD actions vs negative alternatives.
- Keep the safety layer and arc/local-target executor outside SAC.
- Add a result figure that shows positive executed path vs counterfactual bad local target choices.

## Risks / open questions

- Heavy sensor stack: LiDAR, GPS, robot poses, urban navigation stack, BEV labels.
- It is not coverage exploration; it is route-following toward GPS/coarse goals.
- Counterfactual labeling is human-feedback-heavy if used continuously, so for RLxF it should be limited to seed data or offline derived labels.
- Single-timestep reward maps lack memory; even the authors list memory as future work.
