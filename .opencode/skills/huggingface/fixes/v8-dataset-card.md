---
tags:
- sakthai
- house-of-sak
- tool-calling
- function-calling
- augmented
- data-quality
- synthetic
license: apache-2.0
language:
- en
size_categories:
- n<1K
pretty_name: SakThai Combined v8 — Augmented Tool-Calling Dataset
task_categories:
- text-generation
task_ids:
- dialogue-modeling
annotations_creators:
- machine-generated
language_creators:
- found
multilinguality:
- monolingual
source_datasets:
- Nanthasit/sakthai-combined-v7
configs:
- config_name: train
  data_files: data/train.jsonl
---

# SakThai Combined Dataset v8

**538 augmented tool-calling examples** extending v7 with targeted data addressing benchmark gaps.

Part of the [SakThai Model Family Collection](https://huggingface.co/collections/Nanthasit/sakthai-model-family-6a64745450b12d421c1f9f02).

## Dataset Statistics

| Property | Value |
|---|---|
| **Total examples** | 538 |
| **Unique tools** | 12 |
| **Format** | OpenAI chat JSONL |
| **Language** | English |
| **License** | Apache 2.0 |
| **Source** | Generated from v7 + 10 augmentation strategies |

### Category Breakdown

| Category | Count | Description |
|---|---|---|
| Simple (1 tool) | 180 | Single tool-call examples |
| Parallel (2+ tools) | 125 | Multi-tool simultaneous calls |
| Irrelevance (0 tools) | 233 | Model must NOT call tools |
| Multi-turn | 39 | Conversation history tracking |

## Data Format

Same format as v7 — OpenAI chat JSONL with `messages` and `tools`:

```json
{
  "messages": [
    {"role": "user", "content": "Weather in Bangkok?"},
    {"role": "assistant", "tool_calls": [
      {"function": {"name": "get_weather", "arguments": "{\"location\": \"Bangkok\"}"}}
    ]}
  ],
  "tools": [{"type": "function", "function": {...}}]
}
```

## Augmentation Strategies

| # | Strategy | Count | Target |
|---|---|---|---|
| 1 | Arguments normalization (norm() whitespace/case) | 30 | norm() function |
| 2 | Argument type coercion (int/float/string) | 20 | type safety |
| 3 | Parallel multi-tool | 20 | Counter containment |
| 4 | Irrelevance (empty pred_names) | 60 | safety |
| 5 | Hard negatives (selection) | 24 | wrong tool errors |
| 6 | Held-out generalization | 20 | unseen tools |
| 7 | Degenerate prevention | 16 | output quality |
| 8 | Multi-turn context | 15 | conversation tracking |
| 9 | Greedy matching | 15 | one-to-one scoring |
| 10 | Strict accuracy (sel+args) | 20 | combined metrics |

## Usage

```python
from datasets import load_dataset

# Load v8
v8 = load_dataset("Nanthasit/sakthai-combined-v8", split="train")
print(f"Loaded {len(v8)} examples")

# Combine with v7 for training
from datasets import concatenate_datasets
v7 = load_dataset("Nanthasit/sakthai-combined-v7", split="train")
combined = concatenate_datasets([v7, v8])
print(f"Combined: {len(combined)} examples")
```

## Data Quality

- **Schema validated**: All tool calls checked against tool definitions
- **Deduplicated**: 2,042 raw examples collapsed to 538 unique
- **Category balanced**: Covers simple, parallel, and irrelevance
- **Edge cases**: Empty strings, null values, whitespace variants

## Gap Fill Supplement

The `data/gap-fill.jsonl` supplement (478 examples) closes 4 critical gaps identified between v7 and v8:

| Gap | v7 | v8 | Fix |
|---|---|---|---|
| Tool coverage | 86 tools | 12 tools | **+98 examples** covering 50 missing tools |
| Safety refusals | 112 | 0 | **+30 safety guardrail examples** |
| Thai language | 160 | 8 | **+150 Thai tool-calling examples** |
| Multi-turn | 1,287 | 39 | **+200 multi-turn examples** |

## Related

- [v7 (base dataset)](https://huggingface.co/datasets/Nanthasit/sakthai-combined-v7) — 2,424 original examples
- [Benchmark v2](https://huggingface.co/datasets/Nanthasit/sakthai-bench-v2)
- [Irrelevance Supplement](https://huggingface.co/datasets/Nanthasit/sakthai-irrelevance-supplement)
- [SakThai Model Family](https://huggingface.co/collections/Nanthasit/sakthai-model-family)

## Citation

```bibtex
@misc{sakthai-v8-2026,
  author = {Beer (beer-sakthai)},
  title = {SakThai Combined Dataset v8 — Augmented Tool-Calling Data},
  year = {2026},
  publisher = {Hugging Face},
  journal = {Hugging Face Datasets},
  howpublished = {\url{https://huggingface.co/datasets/Nanthasit/sakthai-combined-v8}}
}
```
