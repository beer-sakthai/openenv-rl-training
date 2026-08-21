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
- coder
- code-generation
- merged
- rslor
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
---
# SakThai Plus 1.5B Coder

**Code generation + tool-calling — based on Qwen2.5-Coder-1.5B-Instruct.**

Part of the [SakThai Model Family](https://huggingface.co/collections/Nanthasit/sakthai-model-family).

## Quick Start
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("Nanthasit/sakthai-plus-1.5b-coder", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Nanthasit/sakthai-plus-1.5b-coder")
```

## Links
- [SakThai Plus 1.5B (tool-calling)](https://huggingface.co/Nanthasit/sakthai-plus-1.5b)
- [Coder v1 (sakthai-coder-1.5b)](https://huggingface.co/Nanthasit/sakthai-coder-1.5b)
