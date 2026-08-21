---
name: learning-loop
description: Use when the user mentions learning loop, iterative improvement, feedback loop, continuous improvement, multi-cycle training, or wants to run multiple training iterations to progressively improve model performance. Phase: ITERATE.
---

# SakThai Learning Improvement Loop

```
ITERATION 1     ITERATION 2     ITERATION 3     ITERATION 4     ITERATION 5
───────────     ───────────     ───────────     ───────────     ───────────
┌──────┐        ┌──────┐        ┌──────┐        ┌──────┐        ┌──────┐
│ DATA │        │ DATA │        │ DATA │        │ DATA │        │ DATA │
└──┬───┘        └──┬───┘        └──┬───┘        └──┬───┘        └──┬───┘
   ▼               ▼               ▼               ▼               ▼
┌───────┐        ┌───────┐        ┌───────┐        ┌───────┐        ┌───────┐
│ TRAIN │        │ TRAIN │        │ TRAIN │        │ TRAIN │        │ TRAIN │
└──┬────┘        └──┬────┘        └──┬────┘        └──┬────┘        └──┬────┘
   ▼               ▼               ▼               ▼               ▼
┌──────┐        ┌──────┐        ┌──────┐        ┌──────┐        ┌──────┐
│ EVAL │        │ EVAL │        │ EVAL │        │ EVAL │        │ EVAL │
└──┬───┘        └──┬───┘        └──┬───┘        └──┬───┘        └──┬───┘
   ▼               ▼               ▼               ▼               ▼
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ IMPROVE │────▶│ IMPROVE │────▶│ IMPROVE │────▶│ IMPROVE │────▶│ PUBLISH │
└─────────┘     └─────────┘     └─────────┘     └─────────┘     └─────────┘
```

Each iteration builds on the previous — fix what the eval found, improve the data, retrain, re-evaluate.

## The 5-Iteration Plan

### Iteration 1: Baseline — Merge & Measure v2
```
Goal:   Complete the unfinished v2 model
Data:   sakthai-combined-v7 + irrelevance-supplement (existing)
Train:  Already trained (v2 LoRA exists) — just merge + eval
Eval:   Run on bench-v2
Action: Merge LoRA → model.safetensors → upload → benchmark
```
**Success metric**: Selection accuracy > 48.2% (v1 1.5B baseline)

### Iteration 2: rsLoRA + All-Module Tuning
```
Goal:   Retrain 1.5B with the v2 recipe but fix known issues
Data:   sakthai-combined-v7 + irrelevance-supplement
Train:  rsLoRA enabled, all 7 linear targets, dropout 0.05
        3 epochs, bf16, LR 2e-4
Eval:   Run on bench-v2 + BenchmarkQED AutoE
Action: Compare vs Iteration 1. Did rsLoRA help? Did all-module targeting help?
```
**Success metric**: Selection accuracy > Iteration 1

### Iteration 3: Pure Tool-Calling Data (no chat dilution)
```
Goal:   Fix the "larger models underperform" problem
Data:   Create a filtered dataset — ONLY tool-calling examples
        Remove general chat, keep only simple/parallel/irrelevance
Train:  Same hyperparams as Iteration 2
Eval:   Run on bench-v2
Action: Compare vs Iteration 2. Does removing chat data improve tool-calling?
```
**Success metric**: Selection accuracy > Iteration 2. Ideally > 70%.

### Iteration 4: Data Augmentation + Hard Negatives
```
Goal:   Close the gap to 0.5B (91.2%)
Data:   Augment with:
        - More irrelevance examples (model calls tool when it shouldn't)
        - Edge cases: ambiguous queries, nested tool calls
        - Thai language examples (current v7 has some)
        - Hard negatives: similar function names, overlapping params
Train:  Same hyperparams
Eval:   Run on bench-v2 + held-out tools
Action: Compare vs Iteration 3
```
**Success metric**: Selection accuracy > 85%

### Iteration 5: Production Polish
```
Goal:   Publish the best model yet
Data:   Use best config from Iterations 1-4
Train:  Final training run with optimal hyperparams
Eval:   Full eval: bench-v2 + BenchmarkQED + cross-model comparison
Action: Merge → GGUF (Q4_K_M + Q5_K_M) → model card → Space → deploy
        Update HF Collections, publish benchmark results
```
**Success metric**: Published model with complete card, Space, GGUF, and > 0 downloads

## Eval tracking table

Fill this after each iteration:

| Iteration | Model | Selection | Arguments | Irrelevance | Notes |
|---|---|---|---|---|---|
| 0 (current) | 0.5B-merged | 91.2% | 45.7% | 93.3% | Baseline to beat |
| 0 (current) | 1.5B-merged | 48.2% | — | — | v1 baseline |
| 0 (current) | 7B-merged | 57.0% | — | — | v1 baseline |
| **1** | 1.5B-v2-merged | ? | ? | ? | Merge v2 LoRA + eval |
| **2** | 1.5B-rslora | ? | ? | ? | Retrain with rsLoRA |
| **3** | 1.5B-pure-tc | ? | ? | ? | Tool-calling only |
| **4** | 1.5B-augmented | ? | ? | ? | + hard negatives |
| **5** | 1.5B-final | ? | ? | ? | Production candidate |

## Quick commands by iteration

| Step | Command |
|---|---|
| Check current scores | `/hf-eval` |
| Generate synthetic queries | `/hf-bench` (AutoQ) |
| Run LLM judge | `/hf-bench` (AutoE pairwise) |
| Merge LoRA | `/hf-publish` |
| Create GGUF | `/hf-quantize` |
| Write model card | `/hf-publish` |
| Build demo Space | `/hf-space` |
| Review progress | `/hf-plan` |
