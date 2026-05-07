# Spatially-Enhanced Recurrent Memory for Long-Range Mapless Navigation via End-to-End RL

Source: arXiv `2506.05997`

## Problem and core idea

The paper argues that ordinary recurrent memory is not enough for long-range mapless navigation. LSTMs/GRUs can remember temporal facts, but they do not naturally register observations spatially as the robot moves and turns. The proposed Spatially-Enhanced Recurrent Unit (SRU) adds a learned spatial transformation term to recurrent updates so ego-centric observations can be implicitly aligned across time.

## Method details

- Tests spatial memorization separately from ordinary temporal memorization using a synthetic landmark registration task.
- Adds SRU variants of GRU/LSTM, plus a refined gating version called SRU-Ours.
- Uses an attention-based visual network: pretrained depth encoder, spatial attention compression, proprioception/goal input, SRU recurrent memory, and an MLP action head.
- Trains end-to-end with RL, but uses regularization such as temporally consistent dropout and dense memory learning to prevent early convergence to weak temporal shortcuts.

## Key results

- Reports `23.5%` overall navigation success improvement over standard recurrent units.
- Reports `29.6%` improvement over an explicit-mapping RL baseline and `105.0%` over stacked-observation baselines.
- Standard LSTM policies get trapped in dead ends or re-enter hazards; SRU policies better remember spatially relevant observations after viewpoint changes.

## Relevance to this project

Very high. This is the cleanest support for the hypothesis that our PCVM/PCVM-T failure is not only “not enough context,” but “not the right spatial memory.” The rover memory should encode action/pose-conditioned spatial change, not just a sequence of image embeddings.

## Concrete experiments to run next

- Add an auxiliary next-pose / executed-motion head to PCVM-T.
- Train a simple spatial-registration probe on WASD sequences: can the memory predict whether two frames are a revisit after turns/motion?
- Compare vanilla token history vs action-conditioned memory tokens.
- Add temporally consistent dropout to memory training rather than frame-independent dropout.
- Evaluate memory by revisit/stuck prediction before using it inside SAC.

## Risks / open questions

- The paper is goal-navigation-heavy, not coverage exploration.
- It uses stereo/depth and large-scale simulated RL; our rover has sparse real data and Pi constraints.
- The SRU operation itself may be less useful than the broader lesson: memory must be spatially supervised.
