---
name: bitsandbytes
description: Use when the user mentions quantization, bitsandbytes, 4-bit, 8-bit, NF4, QLoRA, memory optimization, or reducing model size for SakThai models. Phase: TRAIN / DEPLOY.
---

# bitsandbytes — Quantization

## Current use in SakThai
All QLoRA training uses 4-bit NF4 quantization via `BitsAndBytesConfig`:
```python
from transformers import BitsAndBytesConfig
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

## Quantization levels
| Type | Config | VRAM savings | Use case |
|---|---|---|---|
| 4-bit NF4 (QLoRA) | `load_in_4bit=True` | ~4x | Training (current) |
| 8-bit | `load_in_8bit=True` | ~2x | Inference |
| None | — | 1x | Full precision |

## Memory comparison (1.5B)
| Mode | VRAM |
|---|---|
| Full float32 | ~6 GB |
| 8-bit | ~3 GB |
| 4-bit NF4 | ~1.5 GB |

## Installation
```bash
pip install bitsandbytes
```
Already a dependency in the training script's PEP 723 header.

## Inference with quantization
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained(
    "Nanthasit/sakthai-context-1.5b-merged",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    device_map="auto",
)
```

## When NOT to use
- GGUF quantized models (already quantized via llama.cpp)
- CPU-only inference (bitsandbytes requires CUDA)
