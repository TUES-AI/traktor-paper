# FrontierNet: Learning Visual Cues to Explore

Source: arXiv `2501.04597`, repo: `https://github.com/cvg/FrontierNet`

## Problem and core idea

Classical frontier exploration extracts frontiers from dense 3D maps, which makes exploration sensitive to map quality and discards rich RGB cues. FrontierNet instead predicts frontier pixels and information gain directly from posed RGB images augmented with monocular depth priors.

## Method details

- Uses a two-head U-Net-style model.
- Head 1 predicts a frontier distance field in image space.
- Head 2 predicts information gain for frontier regions.
- Ground truth is generated from HM3D voxelized scans: frontier voxels are projected to image pixels, refined with depth discontinuities, and labeled with approximate unknown-volume gain.
- Predicted 2D frontiers are clustered, assigned viewing directions from depth gradients, lifted to sparse 3D candidate frontiers, and tracked in a frontier tree.
- Includes a map-free variant that uses only the frontier tree and the robot's past trajectory as sparse memory.

## Key results

- Reports roughly `15%` improvement in early-stage exploration efficiency.
- With predicted monocular depth, FrontierNet can outperform baselines that use perfect simulator depth for frontier extraction.
- Map-free experiments remain promising and show little revisiting behavior.
- Real-world Spot validation runs at about `5 Hz` with a front RGB camera and shows sim-to-real robustness despite training on renderings.

## Relevance to this project

High. This is the best source for turning “visual novelty” into a trainable exploration target. It suggests that the rover should not rely only on frame-difference novelty; it should learn cues for “this direction likely opens new space.”

## Concrete experiments to run next

- Use Depth-Anything predictions on WASD frames to create approximate depth discontinuity / opening labels.
- Add a lightweight “frontier/open-space likelihood” head to the visual encoder.
- Compare novelty-only vs novelty-plus-frontier score on logged sequences.
- Use the dashboard to visualize predicted frontier pixels over camera frames.
- Maintain a sparse frontier/visited-pose memory rather than a full metric map.

## Risks / open questions

- Training labels depend on privileged 3D scans; our apartment logs do not have this.
- The main evaluation is mapping volume, not embodied coverage with poor odometry.
- The map-free variant is encouraging but not the main full-performance system.
