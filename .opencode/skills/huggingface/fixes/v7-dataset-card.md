---
tags:
- sakthai
- house-of-sak
- tool-calling
- function-calling
- synthetic
- agent-training
- data-quality
- augmented
license: apache-2.0
language:
- en
- th
size_categories:
- 1K<n<10K
pretty_name: SakThai Combined v7 — Tool-Calling Training Dataset
task_categories:
- text-generation
task_ids:
- dialogue-modeling
annotations_creators:
- expert-generated
- machine-generated
language_creators:
- found
multilinguality:
- multilingual
source_datasets:
- original
configs:
- config_name: train
  data_files: data/train.jsonl
- config_name: test
  data_files: data/test.jsonl
---

# SakThai Combined Dataset v7

**Tool-calling, multi-turn, safety, and edge cases — the base training data for the SakThai model family.**

> ⚠️ **v8 recommended**: [sakthai-combined-v8](https://huggingface.co/datasets/Nanthasit/sakthai-combined-v8) extends this dataset with 538 targeted examples addressing benchmark gaps (arguments normalization, parallel calls, hard negatives). For new projects, use v7 + v8 combined.

Part of the [SakThai Model Family Collection](https://huggingface.co/collections/Nanthasit/sakthai-model-family-6a64745450b12d421c1f9f02).

## Dataset Statistics

| Property | Value |
|---|---|
| **Train examples** | 2,309 |
| **Test examples** | 115 |
| **Total** | 2,424 |
| **Format** | OpenAI chat JSONL |
| **Languages** | English, Thai |
| **License** | Apache 2.0 |
| **Tools** | 86 unique function schemas |

### Category Breakdown (train)

| Category | Count | Description |
|---|---|---|
| Tool-calling | ~900 | Single and multi-tool invocations |
| Multi-turn | ~500 | 3+ message exchanges |
| Irrelevance | ~300 | Requests needing no tools |
| Safety | ~200 | Refusal, guardrails, out-of-scope |
| Bilingual (Thai) | ~100 | Thai language tool-calling |

## Data Format

Each example follows the OpenAI chat format:

```json
{
  "messages": [
    {"role": "system", "content": "You are SakThai-Agent..."},
    {"role": "user", "content": "Search for papers on RLHF"},
    {"role": "assistant", "tool_calls": [
      {"function": {"name": "search_papers", "arguments": "{\"query\": \"RLHF\"}"}}
    ]},
    {"role": "tool", "content": "{\"results\": [...]}"},
    {"role": "assistant", "content": "Here's what I found..."}
  ],
  "tools": [
    {"type": "function", "function": {
      "name": "search_papers",
      "description": "Search academic papers",
      "parameters": {"type": "object", "properties": {...}}
    }}
  ]
}
```

### Sample Row

```json
{
  "messages": [
    {"role": "user", "content": "What's the weather in Bangkok?"}
  ],
  "tools": [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get current weather",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
      },
      "required": ["location"]
    }
  }}]
}
```

## Usage

```python
from datasets import load_dataset

# Load training split
train = load_dataset("Nanthasit/sakthai-combined-v7", split="train")
print(f"Train: {len(train)} examples")

# Load test split
test = load_dataset("Nanthasit/sakthai-combined-v7", split="test")
print(f"Test: {len(test)} examples")

# Format for TRL training
def to_text(ex):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    return {"text": tokenizer.apply_chat_template(
        ex["messages"], tools=ex.get("tools"), tokenize=False
    )}
```

## Data Quality

- **Validation**: Each example checked against tool schemas for correctness
- **Coverage**: 86 unique tool schemas across diverse domains
- **Edge cases**: Empty strings, null values, unicode, extreme lengths
- **Irrelevance**: ~300 examples where model must NOT call tools

## Recommended: Combine v7 + v8

v8 adds 538 targeted examples fixing known benchmark gaps. For best results, combine both:

```python
from datasets import load_dataset, concatenate_datasets

v7 = load_dataset("Nanthasit/sakthai-combined-v7", split="train")
v8 = load_dataset("Nanthasit/sakthai-combined-v8", split="train")
combined = concatenate_datasets([v7, v8])
print(f"Combined: {len(combined)} examples")
```

v8 targets:
- **Arguments normalization** — trains norm() whitespace/case handling
- **Parallel calls** — Counter multiset containment
- **Hard negatives** — selection accuracy against similar tools
- **Held-out generalization** — unseen tool schemas
- **Multi-turn context** — conversation tracking
- **Irrelevance detection** — when NOT to call tools

→ [sakthai-combined-v8](https://huggingface.co/datasets/Nanthasit/sakthai-combined-v8)

## Citation

```bibtex
@misc{sakthai-v7-2026,
  author = {Beer (beer-sakthai)},
  title = {SakThai Combined Dataset v7 — Tool-Calling Training Data},
  year = {2026},
  publisher = {Hugging Face},
  journal = {Hugging Face Datasets},
  howpublished = {\url{https://huggingface.co/datasets/Nanthasit/sakthai-combined-v7}}
}
```

## Related

- [v6 (predecessor)](https://huggingface.co/datasets/Nanthasit/sakthai-combined-v6)
- [v8 (augmented extension)](https://huggingface.co/datasets/Nanthasit/sakthai-combined-v8)
- [Benchmark v2](https://huggingface.co/datasets/Nanthasit/sakthai-bench-v2)
- [Irrelevance Supplement](https://huggingface.co/datasets/Nanthasit/sakthai-irrelevance-supplement)
- [SakThai Model Family](https://huggingface.co/collections/Nanthasit/sakthai-model-family)
