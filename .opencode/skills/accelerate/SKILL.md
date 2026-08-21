---
name: accelerate
description: Use when the user mentions Accelerate, multi-GPU, DeepSpeed, FSDP, training speed, distributed training, or optimizing training for any SakThai model size (0.5B, 1.5B, 7B). Phase: TRAIN.
---

# Accelerate — Training Optimization

## How SakThai uses Accelerate
SFTTrainer uses Accelerate internally. Current scripts rely on:
```python
device_map="auto"
gradient_checkpointing=True
gradient_accumulation_steps=8
```

## Explicit config for 7B training
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
mixed_precision: bf16
num_processes: 2
deepspeed_config:
  gradient_accumulation_steps: 8
  gradient_clipping: 1.0
  zero_stage: 2
  offload_optimizer_device: cpu
  offload_param_device: cpu
```
```bash
accelerate launch --config_file accelerate_config.yaml train-sakthai-7b.py
```

## HF Jobs multi-GPU flavors
| Flavor | GPUs | Use for |
|---|---|---|
| `a10g-small` | 1×A10G (24GB) | 1.5B QLoRA |
| `a10g-large` | 4×A10G (24GB) | 7B QLoRA |
| `a100-large` | 8×A100 (80GB) | Large scale |

## Memory estimates
| Config | 0.5B | 1.5B | 7B |
|---|---|---|---|
| Full precision (no QLoRA) | 2 GB | 6 GB | 28 GB |
| 4-bit QLoRA, batch=2 | 1 GB | ~10 GB | ~20 GB |
| + DeepSpeed ZeRO-2 (2 GPU) | — | ~5 GB/GPU | ~12 GB/GPU |
| + ZeRO-3 + CPU offload | — | ~3 GB/GPU | ~8 GB/GPU |

## Local CPU training
The 0.5B CPU script uses Accelerate default mode with float32:
```
uv run train-sakthai-cpu.py
```
No special config needed.
