# PoliFormer: Scaling On-Policy RL with Transformers Results in Masterful Navigators

Source: arXiv `2406.20083`, CoRL / PMLR 2025

## Problem and core idea

PoliFormer asks whether large transformer policies can be trained end-to-end with on-policy RL for RGB-only indoor navigation. The answer is yes, but only at very large simulation scale.

## Method details

- RGB-only policy with frozen DINOv2 visual backbone.
- Goal-conditioned transformer state encoder plus causal transformer decoder for temporal memory.
- Uses KV-cache to make transformer rollout collection feasible.
- Trained with DD-PPO, hundreds of parallel rollouts, large batch sizes, and hundreds of millions of simulator interactions.
- Runs on two embodiments: LoCoBot and Stretch RE-1.

## Key results

- `85.5%` success on CHORES-S, a `28.5%` absolute improvement over prior SOTA.
- Improves over baselines on ProcTHOR, ArchitecTHOR, and AI2-iTHOR.
- Zero-shot sim-to-real gains: `+13.3%` for LoCoBot and `+33.3%` for Stretch RE-1.
- Failure analysis still reports limited memory: after more than about four rooms, the agent may revisit previously explored rooms.

## Relevance to this project

Medium-high as architecture evidence, low as a training recipe. It supports frozen visual foundation features plus causal memory, but the training scale is completely mismatched to a real rover project.

## Concrete experiments to run next

- Use frozen MobileNet/DINO-like features as a baseline against our learned contrastive encoder.
- Keep a causal token-memory model, but train it offline on logged rover data before online SAC.
- Use KV-cache or short summary tokens if PCVM-T runs on the Pi.
- Include PoliFormer as evidence that transformer memory can work, but argue our contribution is low-data real-world adaptation.

## Risks / open questions

- Requires hundreds of millions of simulation interactions and multi-GPU rollout infrastructure.
- ObjectNav is not coverage.
- The remaining room-revisit failure is exactly the kind of long-memory issue our rover must avoid.
