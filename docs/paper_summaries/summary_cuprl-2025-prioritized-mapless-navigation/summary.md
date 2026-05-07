# CUPRL: Image-based Mapless Navigation of a Hybrid Aerial-Underwater Vehicle using Prioritized Deep Reinforcement Learning

Source: Journal of Intelligent & Robotic Systems 2025, DOI `10.1007/s10846-024-02206-z`, repo: `https://github.com/dranaju/cuprl_navigation`

## Problem and core idea

CUPRL addresses pixel-based mapless navigation for a hybrid aerial-underwater vehicle. The core idea is to train a representation from RGB and depth together, but deploy using RGB only.

## Method details

- Extends CURL-style contrastive learning with prioritized replay and SAC.
- Stacks three temporal RGB frames for the query encoder and the corresponding depth maps for the key encoder.
- Contrastive loss aligns RGB and depth representations.
- SAC actor/critic use the learned RGB-side latent plus target/pose information.
- Outputs continuous linear/angular velocities for 3D target navigation.
- Reward: arrival if within `40 cm`; collision if minimum distance is below `62 cm`; small shaping reward for progress toward target.

## Key results

- In air-water transition experiments, CUPRL reaches `100%` success in both evaluation scenarios.
- Water-air transition remains weak: about `24.5%` in one scenario and `14.6%` in another.
- Classical RGB CURL and SAC-CNN baselines often fail or hover; contrastive RGB-depth learning is the main win.
- The authors explicitly note real-world deployment remains future work.

## Relevance to this project

Medium-high. The strongest transferable idea is privileged training: use depth or other sensors/teachers during representation learning, but keep runtime input lean. This fits our Pi constraint and the idea of using Depth-Anything/dashboard supervision offline.

## Concrete experiments to run next

- Train the rover encoder with RGB frame input and depth-teacher targets from Depth-Anything.
- Add contrastive RGB-depth or RGB-motion representation loss on WASD recordings.
- Use prioritized sampling for rare events: recovery, near obstacle, stuck, sharp turn, and high novelty.
- Do not copy their direct velocity output as our final policy interface; keep local target + safety executor.

## Risks / open questions

- Sim-only and vehicle/domain are far from our rover.
- Target-reaching is not coverage.
- Uses target and pose information we do not have in the final no-map setting.
- Water-air failures show that even privileged contrastive learning does not solve hard domain transitions automatically.
