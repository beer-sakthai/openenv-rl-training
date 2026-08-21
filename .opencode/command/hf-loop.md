---
description: Run the 5-iteration SakThai learning improvement loop. Each iteration: data → train → eval → improve → next. Replaces previous iteration's model with an improved version.
agent: general
---

Guide the user through the 5-iteration SakThai learning improvement loop. Read the learning-loop skill (`.opencode/skills/learning-loop/SKILL.md`) for the full plan.

## Determine current iteration

Ask: **"Which iteration are you on?"**

- **Iteration 1**: Baseline — merge existing v2 LoRA → eval → establish baseline
- **Iteration 2**: rsLoRA + all-module retrain → compare vs Iteration 1
- **Iteration 3**: Pure tool-calling data → compare vs Iteration 2
- **Iteration 4**: Data augmentation + hard negatives → compare vs Iteration 3
- **Iteration 5**: Production polish → final merge → GGUF → card → Space → deploy

Or **start from scratch** and walk through all 5 in sequence.

## Per-iteration workflow

For EACH iteration, follow this checklist:

### 1. DATA — Prepare training data
```
Previous iteration's eval showed weaknesses in:
- [ ] Which categories underperformed? (simple/parallel/irrelevance)
- [ ] What errors did the model make? (wrong tool, no tool, hallucinated params)
- [ ] Iteration-specific data changes needed
```

### 2. TRAIN — Fine-tune the model
```bash
hf jobs uv run --flavor a10g-small --timeout 6h --secrets HF_TOKEN train-sakthai-1.5b-v2.py
```
- Update script with any hyperparam changes
- Verify push to Hub after training

### 3. EVAL — Run evaluation
- Run `/hf-eval` for BFCL-style scores
- Run `/hf-bench` for BenchmarkQED LLM judge scores
- Fill the tracking table from the learning-loop skill

### 4. IMPROVE — Decide what to fix next
- Compare scores vs previous iteration
- If target met → move to next iteration
- If not → analyze error patterns → adjust data → retry this iteration

### 5. REPLACE — Tag and archive
- Tag the iteration: `v1.5b-iter1`, `v1.5b-iter2`, etc.
- Update model card with iteration number and results
- Previous iteration model remains on Hub for comparison

## After all 5 iterations

1. Run `/hf-publish` to write final production model card
2. Run `/hf-quantize` to create GGUF (Q4_K_M + Q5_K_M)
3. Run `/hf-space` to build interactive demo
4. Run `/hf-deploy` to set up Inference Endpoint
5. Run `/hf-plan` for next-generation improvements

Return a structured walkthrough for the current iteration with exact commands.
