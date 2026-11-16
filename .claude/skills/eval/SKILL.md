---
name: eval
description: Evaluation procedures, benchmark protocols, and scoring metrics for SakThai tool-calling models and OpenEnv RL agents. Use when running evaluations (sakthai-bench-v2, PEFT adapters, cross-model comparisons), calculating selection/argument/strict metrics, inspecting baseline policies, or generating .eval_results/ badge files.
---

# SakThai & OpenEnv Evaluation

## Evaluation Workspaces & Scripts

| Workspace | Script | Target | Metrics |
|---|---|---|---|
| `sakthai-sft-training/` | `eval-cross-model.py` | 0.5B / 1.5B / 7B merged & LoRA | Selection, Argument, Strict, Irrelevance |
| `sakthai-agentic-eval-train/` | `scripts/eval_bench_peft.py` | `sakthai-bench-v2` test split | Selection, Arguments, Strict, Held-out |
| `sakthai-agentic-eval-train/` | `eval_sakthai_bench_colab.ipynb` | Colab T4 interactive eval | Cell-by-cell benchmark scores |
| `openenv-custom-training/` | `evaluate_baselines.py` | RL environments (Guess, Catch) | Win rate, steps, GRPO group reward variance |

## Core Evaluation Metrics

BFCL-style function calling evaluation evaluates three distinct criteria:

1. **Selection Accuracy**: Did the model select the exact required function name(s)?
   - Multiset comparison accounting for parallel calls (`collections.Counter`).
2. **Arguments Accuracy**: For correctly selected functions, did the argument JSON match the gold standard?
   - Normalized comparison ignoring trivial whitespace/ordering discrepancies (`norm(args)`).
3. **Strict Accuracy**: Selection Accuracy AND Arguments Accuracy both match 100%.
4. **Irrelevance Accuracy**: When no tool call was required (`tool_calls: []`), did the model refrain from calling any tools?

## Benchmark Categories (sakthai-bench-v2)

- **simple** (150 examples): Single tool call with arguments.
- **parallel** (150 examples): Multiple tool calls in a single turn.
- **irrelevance_tools** (150 examples): Tools available, but query is irrelevant or conversational.
- **irrelevance_no_tools** (50 examples): No tools provided, pure conversation.
- **held-out** (100 examples): Tools not seen during training.

## Running Benchmark Evaluations

### Cross-Model Comparison
```bash
# Evaluate multiple models across sakthai-bench-v2
MODELS="Nanthasit/sakthai-context-0.5b-merged,Nanthasit/sakthai-plus-1.5b-lora" uv run eval-cross-model.py
```

### PEFT Direct Evaluation
```bash
# Evaluate a PEFT adapter directly without prior weight merging
uv run --with peft --with transformers --with datasets \
  python3 sakthai-agentic-eval-train/scripts/eval_bench_peft.py \
    --model-id Nanthasit/sakthai-context-7b-tools \
    --base-model Qwen/Qwen2.5-7B-Instruct
```

### RL Baseline Evaluation
```bash
# Verify environment mechanics and baseline reward spreads
uv run --with datasets --with pydantic \
  python3 openenv-custom-training/evaluate_baselines.py
```

## Benchmark Badge Publication (.eval_results/)

Publish evaluation summaries to the model repository root under `.eval_results/sakthai-bench-v2.yaml`:
```yaml
task:
  - text-generation
dataset:
  - sakthai-bench-v2
metrics:
  - selection: 91.8
  - arguments: 45.7
  - strict: 45.7
```
This format is automatically indexed by Hugging Face Hub for model badge rendering.
