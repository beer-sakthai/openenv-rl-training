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

# SakThai Context 1.5B — Merged v2

**Improved tool-calling with rsLoRA + all 7 module targets.**

Part of the [SakThai Model Family](https://huggingface.co/collections/Nanthasit/sakthai-model-family-6a64745450b12d421c1f9f02).

## Improvements over v1
- **rsLoRA enabled**: better scaling with rank
- **All 7 linear modules targeted**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Lower dropout**: 0.05 (vs 0.1 in v1)
- **Latest training data**: sakthai-combined-v7 + v8 + v9 (3,659 combined examples)

## Quick Start
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("Nanthasit/sakthai-plus-1.5b", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Nanthasit/sakthai-plus-1.5b")
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
