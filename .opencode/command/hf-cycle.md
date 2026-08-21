---
description: Walk through the complete 11-phase SakThai development cycle — Data → Train → Eval → Error Analysis → Publish → Deploy → Discoverability → Monitor → Profile → Troubleshoot → Iterate.
agent: general
---

Guide the user through the full SakThai development cycle. Read the workflow skill (`.opencode/skills/workflow/SKILL.md`) for the complete 221-line reference.

Ask: **"Which phase are you in?"**

1. **DATA** — dataset format, augmentation, versioning
2. **TRAIN** — QLoRA fine-tuning (v2 config: rsLoRA + all 7 modules)
3. **EVAL** — BFCL bench-v2 + BenchmarkQED + cross-model
4. **ERROR ANALYSIS** — classify failures, targeted fixes
5. **PUBLISH** — merge LoRA → GGUF → model card → collection
6. **DEPLOY** — Gradio Space → Inference Endpoint → Ollama
7. **DISCOVERABILITY** — metadata tags, inference widgets, .eval_results/
8. **MONITOR** — downloads, likes, score drift, token audit
9. **PROFILE** — benchmark tok/s, optimize with vLLM/TGI
10. **TROUBLESHOOT** — diagnose errors
11. **ITERATE** — 5-loop to close gap to 91.2%

## Full cycle walkthrough
1. `/hf-explore` — check current state
2. `/hf-dataset` or `/hf-augment` — prepare data
3. `train-sakthai-1.5b-v2.py` on Kaggle T4 (free)
4. `/hf-eval` + `/hf-bench` — two-tier eval
5. `/hf-analyze-errors` — classify failures
6. `/hf-publish` + `/hf-quantize` — publish
7. `/hf-space` + `/hf-deploy` + `/hf-ollama` — deploy
8. `/hf-discover` — optimization playbook
9. `/hf-monitor` + `/hf-security` — track health
10. `/hf-profile` — optimize speed
11. `/hf-loop` — 5-iteration improvement loop

Return a structured walkthrough for their chosen phase with exact commands.
