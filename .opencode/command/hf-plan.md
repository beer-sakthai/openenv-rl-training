---
description: Analyze gaps across the full SakThai family (12 models, 8 datasets, 3 spaces) and create a prioritized ecosystem roadmap.
agent: general
---

Fetch all Nanthasit repos via API, then produce a comprehensive ecosystem plan covering the full SakThai family (not just 1.5B).

Key known issues to address:
1. **Benchmark inconsistency**: 0.5B at 91.2% vs 1.5B at 48.2% — scores may not be comparable
2. **sakthai-plus-1.5b-lora has no weights** — mergekit merge not run yet
3. **All 3 Spaces are static HTML** — no interactive demos
4. **No GGUF for 7B** — 1.5B GGUF is the most-downloaded artifact
5. **Datasets have 0 downloads** — v7, bench-v2, irrelevance supplement are invisible
6. **Only 1 like across all repos** — no community engagement

Include:
- Current state table
- Gap analysis (model cards, eval results, GGUF, demo Spaces, dataset cards, discovery)
- Prioritized roadmap from quick wins to long-term
- Concrete next actions with specific commands
