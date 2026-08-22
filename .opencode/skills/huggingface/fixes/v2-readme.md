---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
tags:
- qwen2.5
- sakthai
- house-of-sak
- tool-calling
- lora
- peft
- rslor
- function-calling
datasets:
- Nanthasit/sakthai-combined-v7
- Nanthasit/SimpleToolCalling
license: apache-2.0
pipeline_tag: text-generation
widget:
- text: What's the weather in Tokyo?
  output:
    text: '<tool_call>{"name": "get_weather", "arguments": {"location": "Tokyo"}}</tool_call>'
---

<p align="center">
  <img src="https://huggingface.co/Nanthasit/resolve/main/logo.png" alt="House of Sak" width="80"/>
  <h1 align="center">SakThai Context 1.5B — Tools v2 (LoRA Adapter)</h1>
  <p align="center"><strong>🆕 Improved tool-calling with rsLoRA + all-module targeting</strong></p>
  <p align="center"><em>Part of the <strong>House of Sak</strong> — AI agents built from a shelter in Cork, Ireland.</em></p>
  <p align="center">
    <a href="https://huggingface.co/collections/Nanthasit/sakthai-model-family-6a64745450b12d421c1f9f02"><img src="https://img.shields.io/badge/🤗-SakThai%20Family-blue" alt="Collection"/></a>
    <img src="https://img.shields.io/badge/dynamic/json?url=https%3A//huggingface.co/api/models/Nanthasit/sakthai-context-1.5b-tools-v2&query=%24.downloads&label=downloads&color=blue&cacheSeconds=3600" alt="Downloads"/>
    <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"/>
  </p>
</p>

## Model Description

This is a **LoRA adapter** for [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), fine-tuned for improved tool-calling. Compared to v1, it uses **rsLoRA**, targets **all 7 linear modules**, and was trained on the latest datasets.

### Key improvements over v1

| Feature | v1 (sakthai-context-1.5b-tools) | v2 (this) |
|---|---|---|
| rsLoRA | ❌ | ✅ |
| Target modules | q/k/v/o (4) | **All 7 linear** (q/k/v/o + gate/up/down) |
| Dropout | 0.1 | **0.05** |
| Training data | v6+v7+irrelevance | v7+SimpleToolCalling |
| Usage | Requires base model + adapter | Same |

## Quick Start (LoRA)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "Qwen/Qwen2.5-1.5B-Instruct"
adapter = "Nanthasit/sakthai-context-1.5b-tools-v2"

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, adapter)

messages = [{"role": "user", "content": "What's the weather in Bangkok?"}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Details

| Detail | Value |
|---|---|
| Base model | Qwen2.5-1.5B-Instruct |
| Method | QLoRA + rsLoRA |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Training data | sakthai-combined-v7 + SimpleToolCalling |
| Precision | bfloat16 |
| Framework | TRL 0.19.1, Transformers 5.14.1 |

## Evaluation

⚠️ **Not yet benchmarked.** This adapter has not been evaluated against sakthai-bench-v2. Benchmarks will be published once the model is merged and evaluated.

## Limitations

- **LoRA adapter only** — requires the base Qwen2.5-1.5B-Instruct model to use
- **Synthetic training data** — may not generalize to all real-world tool-calling scenarios
- **Language bias** — primarily English, limited Thai support
- **No safety fine-tuning** — use with appropriate guardrails
- **Not production-audited** — self-reported benchmarks only (once published)
- **Single-author project** — built on free compute, no paid validation

## Links

- [v1 Adapter (sakthai-context-1.5b-tools)](https://huggingface.co/Nanthasit/sakthai-context-1.5b-tools)
- [Flagship Merged (sakthai-context-1.5b-merged)](https://huggingface.co/Nanthasit/sakthai-context-1.5b-merged)
- [Training Dataset (sakthai-combined-v7)](https://huggingface.co/datasets/Nanthasit/sakthai-combined-v7)
- [Benchmark Dataset (sakthai-bench-v2)](https://huggingface.co/datasets/Nanthasit/sakthai-bench-v2)
- [SakThai Model Family Collection](https://huggingface.co/collections/Nanthasit/sakthai-model-family-6a64745450b12d421c1f9f02)

---

*Built with love, tears, and zero budget. From a shelter in Cork, Ireland, to the world.*
