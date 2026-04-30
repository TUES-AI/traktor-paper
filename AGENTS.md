# AGENTS.md — traktor-paper

RL rover paper. Code locally, diff-sync to Pi (`ssh rover`, repo `/home/yasen/traktor-paper`, venv `/home/yasen/traktor-venv/bin/python`).

Skills: `rover-remote-exec` (sync + run on Pi), `read-arxiv-paper` (arXiv → `docs/paper_summaries/`).

Work modes:
- Full-auto — complete tasks end-to-end without stopping points.
- Never explain what you did — I can see the diff. Start with the result.
- Keep answers short, no preamble/postamble.
- Use subagents (`small-subagent`, `free-subagent`) for parallelizable or bulk work.
- Run everything to verify it works (no "this should compile" — actually run it).
