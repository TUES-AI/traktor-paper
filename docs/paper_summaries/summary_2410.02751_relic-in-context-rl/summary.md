# ReLIC: A Recipe for 64k Steps of In-Context RL for Embodied AI

Source: arXiv `2410.02751`

## Problem and core idea

ReLIC studies how an embodied agent can adapt within a long history of its own experience. It scales transformer-based in-context RL to tens of thousands of visual navigation steps by changing the RL update scheme and attention mechanics.

## Method details

- Uses PPO for visual object navigation trials across repeated episodes in the same home layout.
- Introduces partial updates: update the policy multiple times inside a long rollout instead of waiting for the full context.
- Introduces Sink-KV / attention-sink variants to stabilize long-context transformer attention.
- Trains with long context and evaluates in unseen scenes where more in-context episodes should improve navigation.

## Key results

- On the main visual task, success improves from about `23%` to `43%` after 15 in-context episodes.
- Closest baseline reaches about `22%` success while ReLIC reaches `43%`.
- Trained with 4k context can generalize to 32k inference context; paper also demonstrates 64k training context.
- Partial updates and Sink-KV are both critical; without them, learning is much weaker or unstable.

## Relevance to this project

Medium. It proves long history can matter and can be trained by RL, but the scale is far beyond our real rover. The direct lesson is not “use 64k context”; it is “if memory is long, the training algorithm must explicitly support credit assignment and stable attention.”

## Concrete experiments to run next

- Do not increase PCVM-T context without a memory training strategy.
- If using longer histories offline, train on chunks with intermediate auxiliary losses rather than sparse rollout reward only.
- Try attention sinks or explicit null-memory tokens if PCVM-T attention becomes unstable.
- Evaluate whether demonstrations in context improve next-action prediction on WASD logs.

## Risks / open questions

- Requires massive RL training scale.
- ObjectNav with simulator success labels is not real-world coverage.
- Not suitable for direct Pi deployment without compression.
