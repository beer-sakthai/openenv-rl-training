---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- qwen2.5
- sakthai
- house-of-sak
- tool-calling
- conversational
- function-calling
- merged
- rslor
datasets:
- Nanthasit/sakthai-combined-v7
- Nanthasit/sakthai-combined-v8
base_model: Qwen/Qwen2.5-1.5B-Instruct
widget:
- text: What's the weather in Tokyo?
  output:
    text: '<tool_call>{"name": "get_weather", "arguments": {"location": "Tokyo"}}</tool_call>'
- text: Who wrote Romeo and Juliet?
  output:
    text: William Shakespeare wrote Romeo and Juliet.
---

<p align="center">
  <img src="https://huggingface.co/Nanthasit/resolve/main/logo.png" alt="House of Sak" width="80"/>
  <h1 align="center">SakThai Context 1.5B — Merged v2</h1>
  <p align="center"><strong>🆕 Improved tool-calling with rsLoRA + all 7 module targets</strong></p>
  <p align="center"><em>Part of the <strong>House of Sak</strong> — AI agents built from a shelter in Cork, Ireland.</em></p>
  <p align="center">
    <a href="https://huggingface.co/collections/Nanthasit/sakthai-model-family-6a64745450b12d421c1f9f02"><img src="https://img.shields.io/badge/🤗-SakThai%20Family-blue" alt="Collection"/></a>
    <img src="https://img.shields.io/badge/dynamic/json?url=https%3A//huggingface.co/api/models/Nanthasit/sakthai-context-1.5b-merged-v2&query=%24.downloads&label=downloads&color=blue&cacheSeconds=3600" alt="Downloads"/>
    <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"/>
  </p>
</p>

## Model Description

This is the **merged version** of `Nanthasit/sakthai-plus-1.5b-lora` adapter on top of [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). Compared to v1, it uses **rsLoRA**, targets **all 7 linear modules**, and was trained on the latest datasets.

### Key improvements over v1

| Feature | v1 (sakthai-context-1.5b-merged) | v2 (this) |
|---|---|---|
| rsLoRA | ❌ | ✅ |
| Target modules | q/k/v/o (4) | **All 7 linear** (q/k/v/o + gate/up/down) |
| Dropout | 0.1 | **0.05** |
| Training data | v6+v7+irrelevance | sakthai-combined-v7 + v8 + v9 |
| Usage | Merged model | **Merged model** |


## Quick Start
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("Nanthasit/sakthai-context-1.5b-merged-v2", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Nanthasit/sakthai-context-1.5b-merged-v2")
messages = [{"role": "user", "content": "What's the weather in Bangkok?"}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Evaluation
⚠️ Not yet benchmarked against sakthai-bench-v2. Benchmarks will be published after evaluation on HF Jobs GPU.

## Training Data
This model was fine-tuned on the latest SakThai datasets:
- [v7](https://huggingface.co/datasets/Nanthasit/sakthai-combined-v7) (2,424 examples)
- [v8](https://huggingface.co/datasets/Nanthasit/sakthai-combined-v8) (+538 augmented examples)

## Links
- [v1 Merged (flagship)](https://huggingface.co/Nanthasit/sakthai-context-1.5b-merged)
- [v2 LoRA Adapter](https://huggingface.co/Nanthasit/sakthai-plus-1.5b-lora)
- [Benchmark Dataset](https://huggingface.co/datasets/Nanthasit/sakthai-bench-v2)
- [SakThai Model Family Collection](https://huggingface.co/collections/Nanthasit/sakthai-model-family-6a64745450b12d421c1f9f02)

## Limitations
- Self-reported benchmarks only (not independently audited)
- Primarily English; limited multilingual support
- Synthetic training data may not cover all real-world edge cases
- Built on free compute — not a commercial product


---

*Built with love, tears, and zero budget. From a shelter in Cork, Ireland, to the world.*