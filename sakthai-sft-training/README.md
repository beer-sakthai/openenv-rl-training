# sakthai-sft-training

**The SFT half of the pipeline.** Supervised fine-tuning (QLoRA on Qwen2.5), the 10-cycle
data-augmentation loop, cross-model / MCP-Bench / lighteval evaluators, ops tooling, and
the 155-row `sakthai-cycle-bench` harness. Was `beer-sakthai/SakThai-Training` on GitHub
before the 2026-08-21 consolidation.

The RL half lives in [`../openenv-custom-training/`](../openenv-custom-training/) and
[`../openenv-multi-catalog-training/`](../openenv-multi-catalog-training/); the eval
harness that connects the two lives in [`../sakthai-agentic-eval-train/`](../sakthai-agentic-eval-train/).

## What runs where

| Script | What it does | Runner |
|---|---|---|
| `train-sakthai-cpu.py` | 20-example CPU smoke test | `uv run train-sakthai-cpu.py` |
| `train-sakthai-1.5b-v2.py` | QLoRA 1.5B on `sakthai-combined-v9`, ~3k examples | `hf jobs uv run --flavor a10g-small --timeout 6h --secrets HF_TOKEN train-sakthai-1.5b-v2.py` |
| `train_qwen.py` | Generic Qwen2.5 QLoRA harness | HF Jobs |
| `eval-cross-model.py` | Bench all Sak models against `sakthai-bench-v3`, push to `Nanthasit/eval_results` | HF Jobs `l4x1` |
| `eval-sakthai-1.5b.py`, `eval_sakthai_15b_v2_fixed.py` | 1.5B-specific evals | HF Jobs `l4x1` |
| `eval-mcp.py`, `eval-light.py` | MCP-Bench + lighteval wrappers | HF Jobs `l4x1` |
| `cycle-100-v*.py` (10 files) | Iterative data-augmentation cycles v2 → v10 | HF Jobs |
| `augment-*.py` | Targeted augmentation (10x, benchmark-targeted, gap-fill) | HF Jobs |
| `audit-and-fix-safety-quality.py` | Data-quality auditor + fixer | Local or HF Jobs |
| `create-{7b-gguf,balanced-benchmark,benchmark-from-data}.py` | Quantization + bench builders | HF Jobs / local |
| `push-all-to-hub.py`, `push-v9-comprehensive.py` | Hub publishing helpers | Local, needs `HF_TOKEN` |
| `sakthai-1.5b-colab.ipynb` | Self-contained SFT notebook | Colab / Kaggle T4 |
| `scripts/ops/{smoke_test,submit_job,weekly_ops_report}.py` | HF-Jobs ops harness | Local |
| `scripts/eval/{contamination_audit,dataset_composition_stats}.py` | Dataset audits | Local |
| `sakthai-cycle-bench/run_eval.py` + `summary.json` | 155-row BFCL bench harness | Local |
| `gap-fill-v8/v8-gap-fill.jsonl` | 478-row augmentation batch (v8) | Data, not a script |

## TRL 0.19 quirks (matter for every training script here)

- `SFTConfig(processing_class=tokenizer, ...)` — the kwarg is `processing_class`, not `tokenizer`.
- `completion_only_loss=True` masks non-assistant tokens via the chat template.
- `hub_strategy="every_save"` protects against HF-Jobs timeouts; set `--timeout` explicitly (default 30 min is too short for training).

## Current benchmark scores (SFT half only)

Cross-linked from `../sakthai-agentic-eval-train/FINDINGS.md`; do not restate numbers here.

| Model | Selection | Arguments |
|---|---|---|
| `sakthai-context-0.5b-merged` | 91.2% | 45.7% |
| `sakthai-context-1.5b-merged` (v1) | 48.2% | — |
| `sakthai-context-7b-merged` | 57.0% | — |
| `sakthai-context-1.5b-merged-v2` | ⏳ pending | ⏳ pending |

## Dataset + model handles on HF Hub

- Training corpora: `Nanthasit/sakthai-combined-v7` (2,424) · `v8` (+538) · `v9` (all-cycle merge, +1,355) · `v12` (retrain-round-v2 rows).
- Bench: `Nanthasit/sakthai-bench-v3` (155 balanced rows, regenerated weekly by `../.github/workflows/monitor.yml`).
- Eval sink: `Nanthasit/eval_results`.
- Runtime pin: `Nanthasit/sakthai-openenv-training` (dataset that pins `openenv==0.4.1` + TRL/transformers versions across both halves of the repo).
- Trained artifacts: `Nanthasit/sakthai-context-{0.5b,1.5b,7b}-tools` (LoRA), `-merged` (bf16), `-*-gguf` (llama.cpp).

## Related skills

Prompt-library workflows for this workspace live under [`../.opencode/skills/`](../.opencode/skills/) —
`cycle-workflow`, `data-augmentation`, `data-quality-auditor`, `training`, `eval`,
`error-analysis`, `cost-optimization`, `hub-api`, `troubleshooting`.
