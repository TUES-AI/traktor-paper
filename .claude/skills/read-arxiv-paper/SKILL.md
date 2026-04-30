---
name: read-arxiv-paper
description: Fetch an arXiv paper from TeX source and write a structured summary focused on the RL rover project. Use when the user provides an arXiv URL/ID.
---

Fetch the TeX (not PDF) with `~/bin/arxiv-src "<url-or-id>"`, read the entrypoint `.tex` + recursively follow `\input`/`\include` files.

Write summary to `docs/paper_summaries/summary_<arxiv-tag>_<short-name>/summary.md` with:
- Problem and core idea
- Method details (short)
- Key results
- Relevance to this project (RL rover, PPO, coverage path planning, Pi constraints)
- Concrete experiments to run next (3-6 bullets)
- Risks / open questions

If the paper has a linked repo, clone to `/tmp/`, extract relevant code, and copy into the summary folder.

> Do not overwrite an existing summary unless asked.
