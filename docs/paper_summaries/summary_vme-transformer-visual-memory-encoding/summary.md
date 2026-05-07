# VME-Transformer: Enhancing Visual Memory Encoding for Navigation in Interactive Environments

Source: IEEE Robotics and Automation Letters 2024, DOI page surfaced via NASA ADS

## Problem and core idea

VME-Transformer targets visual interactive navigation in cluttered environments where the robot may need to remember partial observations and interact with obstacles. The main contribution is a transformer-based visual memory encoder that stores both recent and long-term exploration information.

## Method details

- Uses a Transformer Visual Memory Encoder for history representation.
- Adds explicit next-pose prediction conditioned on the impending action to bootstrap representation learning.
- Regularizes the value function with input perturbations to improve generalization.
- Evaluated in iGibson visual interactive navigation tasks.

## Key results

- Reports superior performance over state-of-the-art visual interactive navigation baselines in iGibson.
- The important qualitative finding is that high-capacity memory benefits from an auxiliary dynamics/pose objective rather than sparse RL reward alone.

## Relevance to this project

Very high for PCVM-T design. The next-pose auxiliary objective is directly transferable and likely more valuable to us than the exact transformer architecture.

## Concrete experiments to run next

- Add an auxiliary head to predict executed delta pose / yaw from current memory and action.
- Use recovery/stuck events as negative examples for the same auxiliary training pass.
- Perturb visual embeddings or drop frames during offline memory training to test robustness.
- Compare PCVM-T with and without next-motion prediction on WASD logs.

## Risks / open questions

- The task is interactive navigation in simulation, not real low-cost coverage.
- Details are less accessible than arXiv-source papers.
- It may depend on simulator pose labels that our rover only approximates.
