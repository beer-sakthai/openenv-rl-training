---
name: training
description: Recipes, configurations, and procedures for fine-tuning and reinforcement learning (SFTTrainer, GRPOTrainer, QLoRA, rsLoRA) across SakThai and OpenEnv workspaces. Use when setting up SFT or RL runs, configuring PEFT target modules, selecting hardware flavors (Local CPU, Kaggle T4, HF Jobs), setting hyperparams, or pushing model weights and adapters.
---

# SakThai & OpenEnv Training

## Architectures & Workspaces

The repository supports two primary training paradigms:
1. **Supervised Fine-Tuning (SFT)** in `sakthai-sft-training/`:
   - Fine-tunes Qwen2.5 base models (0.5B, 1.5B, 7B) on tool-calling conversational datasets using TRL `SFTTrainer`.
2. **Reinforcement Learning with GRPO** in `openenv-custom-training/` and `openenv-multi-catalog-training/`:
   - Trains policy models against simulated or containerized environments (OpenEnv tasks, Catch, Sudoku, Echo, BrowserGym) using TRL `GRPOTrainer(environment_factory=...)`.

## The Production SFT Recipe (v2 Baseline)

Empirical evaluation established the critical factors separating high-performing models:
- **All 7 linear projection modules** must be targeted by LoRA:
  `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`.
- **rsLoRA** (`use_rslora=True`): Stabilizes scaling as rank varies.
- **Completion-only loss masking**: Non-assistant tokens are masked out during loss calculation to prevent language modeling distraction on system prompts and tools.
- **Pure tool-calling data**: Avoid general chat dilution that degrades tool selection precision.

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_rslora=True,
)
```

## TRL SFTTrainer API Conventions

- Use `SFTConfig` rather than deprecated `TrainingArguments`.
- Pass tokenizer via `processing_class=tokenizer` (TRL >= 0.12 convention).
- Pass tools directly through `apply_chat_template(messages, tools=tools, ...)`.

## Hyperparameters by Compute Target

| Model Size | Platform | Precision | Epochs | Batch / Grad Accum | Max Seq | Approx VRAM | Cost |
|---|---|---|---|---|---|---|---|
| **0.5B (Debug)** | Local CPU | fp32 | 1 | 1 / 1 | 512–1024 | ~2 GB RAM | $0 |
| **1.5B** | Kaggle T4 / HF A10G | bf16 (QLoRA NF4) | 3 | 2 / 8 | 2048 | ~10 GB | $0 / ~$1/hr |
| **7B** | HF Jobs A10G / A100 | bf16 (QLoRA NF4) | 3 | 1 / 8 | 2048 | ~18–22 GB | ~$1–$3/hr |

## OpenEnv RL Training (GRPOTrainer)

For RL training in `openenv-custom-training/train.py`:
- Use `GRPOConfig` with `per_device_train_batch_size`, `num_generations=4`, `max_completion_length=128`.
- Ensure environment factories implement standard OpenEnv step / reset contracts.
- Wrap step outputs in conversational message structures conforming to model chat templates.

## Model Publication Checklist

1. Save PEFT LoRA adapter: `model.save_pretrained("./lora-adapter")` -> push to `<repo>-lora`.
2. Merge LoRA adapter into base model: `model.merge_and_unload()` -> push to `<repo>-merged`.
3. Push tokenizer and chat template to both adapter and merged repositories.
4. Set `hub_strategy="every_save"` during training to safeguard against timeout loss.
