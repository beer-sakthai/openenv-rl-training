---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- qwen2.5
- sakthai
- plus
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
---

# SakThai Plus 1.5B

**Next-generation tool-calling model — rsLoRA + all 7 module targets.**

Part of the [SakThai Model Family](https://huggingface.co/collections/Nanthasit/sakthai-model-family-6a64745450b12d421c1f9f02).

## Improvements over v1 (sakthai-context-1.5b-merged)
- **rsLoRA enabled**: better scaling with rank
- **All 7 linear modules targeted**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Lower dropout**: 0.05 (vs 0.1 in v1)
- **Latest training data**: v7 + v8 combined (2,962 examples)

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

## Links
- [v1 (sakthai-context-1.5b-merged)](https://huggingface.co/Nanthasit/sakthai-context-1.5b-merged)
- [Plus LoRA Adapter](https://huggingface.co/Nanthasit/sakthai-plus-1.5b-lora)
- [Training Dataset v7](https://huggingface.co/datasets/Nanthasit/sakthai-combined-v7)
- [Training Dataset v8](https://huggingface.co/datasets/Nanthasit/sakthai-combined-v8)
- [Benchmark](https://huggingface.co/datasets/Nanthasit/sakthai-bench-v2)
