# TANGO: Training-free Embodied AI Agents for Open-world Tasks

Source: arXiv `2412.10402`, CVPR 2025

## Problem and core idea

TANGO is a training-free modular embodied agent. Instead of training a new policy for each task, it asks an LLM to compose programs from prebuilt perception, navigation, exploration, memory, and question-answering modules.

## Method details

- Uses a pretrained PointGoal navigation policy as the low-level navigation primitive.
- Uses a VLFM-style exploration policy with depth-derived occupancy/frontiers and language-grounded value maps.
- Adds a memory feature map so previously seen target-related regions can be reused for sequential goals.
- Uses object detection/classification modules such as OwlV2, DETR, CLIP classifiers, BLIP2, and image matching.
- LLM generates readable pseudo-code/programs from few in-context examples.

## Key results

- Open-set ObjectNav validation-unseen: `35.5%` SR and `19.5%` SPL, near VLFM.
- GOAT-Bench: reports `+2.6%` success over prior SOTA and second-best efficiency.
- OpenEQA: ranks second among zero-shot approaches, close to best, but humans remain far ahead at about `85%`.
- Failure analysis: detector failures dominate; LLM program errors happen around `18%` of failures.

## Relevance to this project

Low-medium for control, medium for research framing. The useful part is modularity and explainable failure tracing, not the LLM planner itself. For our rover, VLMs/LLMs should label data or help dashboards, not drive motors.

## Concrete experiments to run next

- Use VLM/LLM modules offline to annotate WASD frames with room/object/place cues.
- Keep the runtime rover stack small: local target policy + safety + executor.
- Borrow the “program trace” idea for dashboard explanations of why a local target was chosen.
- Treat memory as a feature map/summary bank, but not as a full metric map.

## Risks / open questions

- Depends on depth/GPS/compass/PointGoal assumptions that we do not have.
- Not a learned real-world RL method.
- LLM code generation introduces non-determinism and failure modes inappropriate for low-level safety.
