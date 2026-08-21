---
name: training
description: Use when the user mentions training, fine-tuning, QLoRA, LoRA, rsLoRA, HF Jobs, Kaggle training, SFTTrainer, TRL, mergekit, or running train-*.py scripts. Phase: TRAIN.
---

# SakThai Training

Part of the **TRAIN phase** of the SakThai development cycle. See also: `cost-optimization` skill (cheapest hardware), `data-format` skill (dataset structure).

## Scripts
| File | Model | Data | Hardware | Purpose |
|---|---|---|---|---|
| `train-sakthai-1.5b-v2.py` | Qwen2.5-1.5B-Instruct | v7 + supplement | Kaggle T4 ($0) or HF Jobs A10G ($1/hr) | **Current best** |
| `train-sakthai-cpu.py` | Qwen2.5-0.5B-Instruct | v7 + supplement (20 rows) | Local CPU | Test/debug |

## Recommended config (from v2 production run)
This is the config that produces the best results — use as starting point for all new training:
```python
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # ALL 7 linear
    use_rslora=True,  # key improvement over v1
)
```
**Why this matters**: The 0.5B model (91.2% selection) uses **all 7 modules** + rsLoRA. The 1.5B v1 (48.2%) only targets 4 modules with no rsLoRA. This single change is the biggest lever for improvement.

## Training method
- **QLoRA**: 4-bit NF4 quantization, bfloat16 compute, double quant
- **rsLoRA**: rank=16, alpha=32, dropout=0.05, all 7 linear targets
- **Completion-only loss**: `completion_only_loss=True` → masks non-assistant tokens
- **Dataset**: `apply_chat_template(msgs, tools=tools, ...)` → `dataset_text_field="text"`

## Dataset loading
```python
main = load_dataset("Nanthasit/sakthai-combined-v7", split="train")
supp = load_dataset("Nanthasit/sakthai-irrelevance-supplement", split="train")
train_data = concatenate_datasets([main.map(to_text), supp.map(to_text)])
eval_data = load_dataset("Nanthasit/sakthai-combined-v7", split="test").map(to_text)
```

## TRL 0.19 API quirks
- Use `SFTConfig` (not `TrainingArguments`)
- `SFTTrainer(processing_class=tokenizer, ...)` — param is `processing_class`, not `tokenizer`
- `completion_only_loss=True` masks non-assistant tokens via chat template

## Hyperparams by model size
| Param | 0.5B (CPU) | 1.5B (Kaggle/HF) | 7B (HF Jobs) |
|---|---|---|---|
| epochs | 1 | 3 | 3 |
| batch size | 1 | 2 × 8 grad accum | 1 × 8 grad accum |
| learning rate | 2e-4 | 2e-4 | 1e-4 |
| max seq len | 1024 | 2048 | 2048 |
| precision | float32 | bf16 | bf16 |
| VRAM | 2 GB | ~10 GB | ~20 GB |
| **Cost/hr** | **$0** | **$0 (Kaggle) / $0.40 (HF T4)** | **$0.40/hr (HF T4)** |

## Cheapest execution path
1. **Merge v2 LoRA**: HF Jobs CPU Basic ($0.01/hr × 0.5h = $0.005)
2. **Retrain 1.5B**: Kaggle T4 (free, 30 hrs/week quota)
3. **Eval**: HF Jobs CPU Basic ($0.01/hr × 0.5h = $0.005)
4. **Full 5-iteration loop**: ~$0.02 total

## Push behavior
1. Adapter → `Nanthasit/sakthai-plus-1.5b-lora`
2. Merge + push merged → `Nanthasit/sakthai-plus-1.5b`
3. Tokenizer pushed to both
4. `hub_strategy="every_save"` protects against timeout data loss

## Reminders
- `HF_TOKEN` required (`--secrets HF_TOKEN`)
- Set `--timeout` explicitly (default 30 min too short)
- 0.5B CPU script uses float32, saves locally only
- For 7B: reduce batch or use DeepSpeed ZeRO-2
