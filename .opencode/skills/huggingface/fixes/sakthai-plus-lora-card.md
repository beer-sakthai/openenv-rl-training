---
license: apache-2.0
library_name: peft
tags:
- qwen2.5
- sakthai
- plus
- tool-calling
- lora
- peft
- rslor
- function-calling
datasets:
- Nanthasit/sakthai-combined-v7
- Nanthasit/sakthai-combined-v8
base_model: Qwen/Qwen2.5-1.5B-Instruct
pipeline_tag: text-generation
---

# SakThai Plus 1.5B — LoRA Adapter

**PEFT LoRA adapter for sakthai-plus-1.5b. Requires the base Qwen2.5-1.5B-Instruct model.**

Part of the [SakThai Model Family](https://huggingface.co/collections/Nanthasit/sakthai-model-family).

## Training config
- rsLoRA: enabled
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj (all 7)
- Rank: 16, Alpha: 32, Dropout: 0.05
- Base: Qwen/Qwen2.5-1.5B-Instruct
- Data: v7 + v8 + irrelevance-supplement

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "Nanthasit/sakthai-plus-1.5b-lora")
merged = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained("Nanthasit/sakthai-plus-1.5b-lora")
merged.push_to_hub("Nanthasit/sakthai-plus-1.5b")
tokenizer.push_to_hub("Nanthasit/sakthai-plus-1.5b")
```
