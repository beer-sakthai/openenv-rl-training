---
name: eval
description: Use when the user mentions evaluation, BFCL, benchmarking, before/after comparison, metrics, benchmark inconsistency, cross-model comparison, .eval_results, or running eval scripts. Phase: EVAL.
---

# SakThai Evaluation

Part of the **EVAL phase** of the SakThai development cycle. See also: `error-analysis` skill (deep-dive on failures), `benchmark-qed` skill (LLM-as-a-judge).

## Scripts
| File | Test data | Models | Metrics |
|---|---|---|---|
| `eval-sakthai-1.5b.py` | `sakthai-combined-v7` test | v1 vs v2 merged | Selection acc |
| `eval_sakthai_15b_v2_fixed.py` | `sakthai-bench-v2` raw JSONL | Any merged | Selection + args + strict |
| `eval-cross-model.py` | `sakthai-bench-v2` raw JSONL | ALL models in one run (default 0.5B/1.5B/7B merged) | Selection + args + strict per category + unified table |

`eval-cross-model.py` is the **canonical cross-model script** — same data, same renderer (`apply_chat_template(tools=...)`), same scorer for every model. Env: `MODELS` (comma-separated), `SAMPLE`, `BATCH`, `MAX_NEW`, `DUMP`, `OUT`. LoRA repos (id contains `lora`) auto-load via peft on top of their base model. Writes per-model `.eval_results/sakthai-bench-v2.yaml` badge files + `.cross-model.yaml` detail files.

## ⚠️ Benchmark data (confirmed from actual runs)

| Model | Selection | Arguments | Irrelevance | Held-out | Notes |
|---|---|---|---|---|---|
| **0.5B-merged** | **91.2%** | 45.7% | 93.3% | 87.8% | uses all 7 modules + prompt-masked loss |
| **0.5B-tools** | **91.8%** | 45.7% | 93.3% | 87.8% | same recipe |
| **1.5B-merged** | **48.2%** | — | — | — | only 4 target modules, chat dilution |
| **7B-merged** | **57.0%** | — | — | — | only 4 target modules, chat dilution |

**Root cause** (from model cards): The 0.5B uses **all 7 linear modules** as LoRA targets + **prompt-masked loss** (completion-only). The 1.5B/7B only target q/k/v/o and include general chat data that dilutes tool-calling precision. The 1.5B and 7B cards honestly removed inflated prior scores — they say "Not independently benchmarked."

**Fix**: Apply the v2 recipe (rsLoRA + all 7 targets + pure tool-calling data) to 1.5B.

## BFCL-style categories
- **simple** — 1 tool call expected
- **parallel** — 2+ tool calls expected
- **irrelevance** — 0 tool calls expected (model should not call any tool)

## Metrics
| Script | Metrics | Details |
|---|---|---|
| `eval-sakthai-1.5b.py` | pass/total/acc% per category | Gold name in prediction |
| `eval_sakthai_15b_v2_fixed.py` | selection, arguments, strict | Normalized JSON comparison |
| `eval_bench.py` (in bench-v2 repo) | All above + held-out + degenerate | Official SakThai scorer |

## .eval_results/ format (preferred over old model-index)
Create `.eval_results/sakthai-bench-v2.yaml` in each model repo:
```yaml
task:
  - text-generation
dataset:
  - sakthai-bench-v2
metrics:
  - selection: 91.2
  - arguments: 45.7
  - strict: 45.7
```
This auto-populates benchmark badges. The old `model-index` YAML in README is deprecated.

## Running cross-model eval
`eval-cross-model.py` handles this in one run (defaults to all three):
```bash
uv run eval-cross-model.py                          # 0.5B + 1.5B + 7B, full 500 rows
MODELS="Nanthasit/sakthai-context-0.5b-merged,Nanthasit/sakthai-plus-1.5b-lora" uv run eval-cross-model.py
```
Benchmark categories in the raw data are `simple` / `parallel` / `irrelevance_tools` / `irrelevance_no_tools`
(150/150/150/50); the script merges the two irrelevance categories for reporting.

## Execution
- No local GPU — HF Jobs GPU (a10g-small, needed for 7B) or Kaggle T4
- Cheap CPU option for 0.5B/1.5B only: `hf jobs uv run --flavor cpu-basic --timeout 3h eval-cross-model.py` (BATCH=1, very slow)
- Publish results as `.eval_results/sakthai-bench-v2.yaml` in each model repo
