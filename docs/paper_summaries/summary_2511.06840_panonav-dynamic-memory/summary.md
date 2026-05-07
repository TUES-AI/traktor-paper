# PanoNav: Mapless Zero-Shot Object Navigation with Panoramic Scene Parsing and Dynamic Memory

Source: arXiv `2511.06840`

## Problem and core idea

PanoNav targets RGB-only, mapless, open-vocabulary ObjectNav. It argues that current mapless VLM/LLM agents get stuck in local loops because they reason from the current observation without enough trajectory memory.

## Method details

- Captures six RGB views at 60-degree intervals to form a panorama.
- Converts views into dot-matrix spatial representations, then uses an MLLM for local directional descriptions and a global scene summary.
- Stores global summaries in a Dynamic Bounded Memory Queue.
- An LLM chooses navigation directions from current local/global descriptions plus memory.
- Uses PixNav as the low-level motion controller.

## Key results

- RGB-only mapless open-vocabulary result: `43.5` SR and `23.7` SPL.
- Beats PixNav (`37.9` SR, `20.5` SPL) and ZSON (`25.5` SR, `12.6` SPL) under similar RGB-only mapless settings.
- Memory-guided variant has up to four times higher success than no-memory in selected deadlock tests.
- Escape rate improves from `32.0%` without memory to `82.0%` with memory.
- Reducing panorama from six views to three drops SR to `19.5%`.

## Relevance to this project

Medium. The MLLM/panorama setup is not practical for the Pi rover, but the bounded memory queue is directly useful: maintain compact summaries of recent places/actions to avoid revisits and loops.

## Concrete experiments to run next

- Implement a non-LLM bounded memory queue over visual cluster summaries and executed local targets.
- Penalize candidate local targets that point back toward recent summary states.
- Add a deadlock metric: repeated cluster/state within K actions with low executed distance.
- Visualize memory queue entries on the dashboard.

## Risks / open questions

- ObjectNav, not coverage.
- Six-view panoramic input is not our hardware.
- MLLM/LLM decision loops are slow, expensive, and brittle for real-time safety.
