---
name: cost-optimization
description: Use when the user mentions cost, budget, HF Jobs credits, Kaggle free tier, GPU pricing, optimization, saving money, or running training on a budget. Phase: TRAIN / EVAL.
---

# Cost Optimization — SakThai Training Budget

## Training cost estimates

| Model | Hardware | Cost/hr | Time | Total cost | Frequency |
|---|---|---|---|---|---|
| 0.5B (CPU test) | Local CPU | $0 | ~5 min | **$0** | Per iteration |
| 1.5B (QLoRA) | HF Jobs A10G-small | ~$1.00 | ~3-4 hrs | **$3-4** | Per iteration |
| 1.5B (QLoRA) | Kaggle T4 | $0 (free) | ~4-5 hrs | **$0** | Limited quota |
| 7B (QLoRA) | HF Jobs A10G-large | ~$2.50 | ~8-12 hrs | **$20-30** | Per iteration |
| 7B (QLoRA) | Kaggle T4 | $0 (free) | ~12-16 hrs | **$0** | Limited quota |
| Eval (all 3 models) | HF Jobs CPU | ~$0.15 | ~1 hr | **$0.15** | Per iteration |

## HF Jobs pricing (updated)

| Flavor | GPU | RAM | Cost/hr |
|---|---|---|---|
| `cpu-basic` | None | 16 GB | **$0.01** |
| `cpu-small` | None | 32 GB | $0.03 |
| `t4-small` | T4 (16 GB) | 15 GB | **$0.40** |
| `a10g-small` | A10G (24 GB) | 30 GB | $1.00 |
| `a10g-large` | 4× A10G (24 GB) | 120 GB | $2.50 |

## 5-Iteration loop total cost estimate (cheapest path)

Use **Kaggle T4 free tier** for training (30 hrs/week quota), **HF Jobs CPU** for eval:

| Iteration | Hardware | Time | Cost |
|---|---|---|---|
| 1. Merge v2 + eval | HF Jobs CPU Basic ($0.01/hr) | 0.5h | **$0.005** |
| 2. Retrain 1.5B | Kaggle T4 (free) | 4h | **$0** |
| 3. Retrain pure TC | Kaggle T4 (free) | 4h | **$0** |
| 4. Retrain augmented | Kaggle T4 (free) | 4h | **$0** |
| 5. Final eval + publish | HF Jobs CPU Basic | 1h | **$0.01** |
| **Total** | | | **~$0.02** |

Scheduled jobs via HF: `hf jobs scheduled uv run @weekly --flavor cpu-basic --timeout 30m eval-sakthai-1.5b.py`

## Cost-saving strategies

### 1. Use Kaggle T4 for training (free)
```
- 30 hrs/week GPU quota
- 1.5B QLoRA takes ~4-5 hrs per run → 6+ runs/week
- Upload script as notebook, install deps, run
- Push to Hub directly from Kaggle
```

### 2. HF Jobs CPU for eval (cheap)
```bash
hf jobs uv run --flavor cpu-small --timeout 2h --secrets HF_TOKEN eval_sakthai_15b_v2_fixed.py
```
CPU eval costs ~$0.15/hr instead of $1.00/hr for GPU.

### 3. Cache models locally
The 1.5B base model is ~3 GB. Download once, reuse across iterations:
```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./cache/qwen-1.5b
```
Then mount cache in HF Jobs:
```bash
hf jobs uv run --flavor a10g-small --volume ./cache:/cache train-sakthai-1.5b-v2.py
```

### 4. Gradient checkpointing (already enabled)
Reduces VRAM by ~30% at ~15% speed cost. Currently on for all training.

### 5. Reduce epochs for ablation
For hyperparameter tests, use 1 epoch instead of 3:
```python
num_train_epochs=1  # was 3
```
Cuts training cost to 1/3.

### 6. Early stopping
```python
SFTConfig(
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_strategy="steps",
    save_steps=50,
    eval_strategy="steps",
    eval_steps=50,
)
```
If eval loss plateaus early, training auto-stops.

## HF Jobs credit balance check
```bash
hf jobs list
hf user info  # shows credits
```
## Budget tracking sheet

| Iter | Date | Model | Hardware | Time | Cost | Runner |
|---|---|---|---|---|---|---|
| 1 | | v2 merge | CPU | | $0.15 | |
| 2 | | 1.5B | Kaggle | | $0 | |
| ... | | | | | | |
