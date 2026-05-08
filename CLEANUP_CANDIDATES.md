# Cleanup candidates

Working list only. Nothing here has been deleted yet.

## Likely safe to delete after quick confirmation

These are old wrappers or notes from directions we have moved past.

- `embedded/scripts/train_real_pcvm_t_sac.sh`
  - Old PCVM-T real-training wrapper. Transformer PCVM is not current main direction.
- `embedded/scripts/train_real_predictive_sac.sh`
  - Generic old predictive/PCVM wrapper using the older local-target defaults; current wrappers are the `*_slow_rlxf.sh` and `*_theta_front_rlxf.sh` scripts.
- `embedded/scripts/train_real_pcvm_m_sac.sh`
  - Old PCVM-M wrapper before slow/theta-front RLxF wrappers.
- `embedded/scripts/run_real_sac_vmm.sh`
  - Older run-only local-target wrapper, likely superseded by direct training wrappers and current theta-front scripts.
- `old.md`
  - Early project scratch notes now superseded by `PROJECT.md`, `PLAN.md`, `docs/real_rover_theta_front_direction.md`, and wiki notes.
- `docs/pi_training_plan.md`
  - Old Pi SAC+RND plan. It assumes older raw motor/action and warm-start ideas that conflict with the current strict RLxF/theta-front direction.
- `results/vmm_training/analysis_20260506_185803.md`
  - Old generated analysis note under `results/`; should not be part of the durable source tree unless that whole run is intentionally preserved.

## Local untracked duplicates that can be removed from workspace

The 200-step run is already preserved under `data/autonomous_runs/pcvm_theta_front_20260507_200step_allrooms/`, so these local `results/` copies can be deleted if not needed for immediate inspection:

- `results/pcvm_theta_front_frames_200/`
- `results/pcvm_theta_front_frames_200_contact_sheet.jpg`
- `results/pcvm_theta_front_rlxf_200.jsonl`

Old local untracked logs, probably safe to delete after checking they are not unique:

- `results/pcvm_m_slow_rlxf_train.jsonl`
- `results/pcvm_slow_rlxf_train.jsonl`
- `results/pcvm_theta_front_rlxf_train.jsonl`
- `results/pcvm_theta_front_rlxf_train_rerun.jsonl`

## Review before deleting

These are not obviously current, but may still be useful as hardware docs or paper build files.

- `embedded/GUIDE_EXECUTOR_CALIBRATION.md`
  - Old guide format `[curvature, horizon, speed]`; maybe obsolete experimentally, but contains useful calibration language.
- `docs/real_rover_theta_front_direction.md`
  - Recent direction note. Keep unless `PLAN.md` becomes the single source of truth.
- `CLAUDE.md`
  - Mostly duplicated by `AGENTS.md`, but may still be used by Claude tooling.
- `paper.pdf`
  - Generated build artifact; delete if we do not want PDFs tracked/kept locally.
- `algorithm.sty`, `algorithmic.sty`, `fancyhdr.sty`, `forloop.sty`, `icml2026.sty`, `icml2026.bst`, `example_paper.bib`
  - Paper template/build support files. Do not delete unless LaTeX build no longer needs them or they are available elsewhere.

## Keep for now

- `PLAN.md`
  - Current working implementation plan.
- `PROJECT.md`
  - Main human project scratchboard.
- `embedded/scripts/train_real_pcvm_theta_front_rlxf.sh`
  - Current best real-rover wrapper.
- `embedded/scripts/train_real_pcvm_m_theta_front_rlxf.sh`
  - Current diagnostic MobileNet theta-front wrapper.
- `embedded/scripts/train_real_pcvm_slow_rlxf.sh`
  - Recent slow local-target comparison wrapper.
- `embedded/scripts/train_real_pcvm_m_slow_rlxf.sh`
  - Recent slow PCVM-M comparison wrapper.
- `docs/paper_summaries/**/summary.md`
  - Paper-reading notes; keep unless a separate archival cleanup is done.
- `data/autonomous_runs/pcvm_theta_front_20260507_200step_allrooms/**`
  - Preserved labeled 200-step all-rooms dataset.
