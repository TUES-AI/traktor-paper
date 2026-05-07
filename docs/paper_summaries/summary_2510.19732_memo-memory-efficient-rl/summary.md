# Memo: Training Memory-Efficient Embodied Agents with RL

Source: arXiv `2510.19732`

## Problem and core idea

Memo addresses the transformer memory explosion problem in long-horizon embodied RL. Instead of keeping every observation token, it periodically compresses segments of experience into learned summary tokens that are reused as memory.

## Method details

- Splits trajectories into segments, commonly `l_seg = 256`.
- Generates a small number of summary tokens per segment, commonly `l_sum = 32`.
- Future segments attend to accumulated summaries plus recent observations.
- Trains summaries end-to-end with the RL objective, while allowing gradients across summary accumulation.
- Randomizes segment lengths during training to avoid overfitting to fixed boundaries.

## Key results

- Matches or outperforms full-context transformer baselines while using about `8x` fewer tokens.
- On extended object navigation, reports about `7.5%` higher success and `2.5%` higher SPL than full-context transformer under the same ReLIC-style training.
- Streaming Memo remains robust when old raw context is discarded; streaming full-context transformers degrade sharply.
- Summary length is sensitive: too few summaries lose information, too many overfit / add noise.

## Relevance to this project

Very high. Memo is the strongest architectural argument against simply making PCVM-T context longer. The rover needs compact memory summaries of recent experience, not unbounded frame history.

## Concrete experiments to run next

- Add summary tokens to PCVM-T: summarize every N frames/actions into a small memory bank.
- Train summary memory with next-motion, revisit, stuck, and novelty prediction losses.
- Test summary sizes such as 4, 8, 16 tokens for Pi-feasible inference.
- Randomize segment boundaries during offline training on WASD logs.
- Compare streaming summary memory vs fixed 16-frame PCVM-T context.

## Risks / open questions

- Still trained in large simulator settings.
- Uses object navigation, not coverage.
- Needs careful tuning; summary length and segment length can cause instability.
