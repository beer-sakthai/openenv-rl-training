---
name: data-format
description: Use when the user mentions dataset format, JSONL structure, to_text conversion, chat template, message schema, tool_call format, data loading, or the specific structure of SakThai datasets. Phase: DATA.
---

# SakThai Data Format

Part of the **DATA phase** of the SakThai development cycle. See also: `data-augmentation` skill (generating new data).

## Common format (all repos)
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "content": "..."}
  ],
  "tools": [{
    "type": "function",
    "function": { "name": "...", "description": "...", "parameters": {...} }
  }]
}
```

## to_text preprocessing
```python
def to_text(ex):
    msgs = ex["messages"]
    tools = ex.get("tools") or None
    text = tokenizer.apply_chat_template(
        msgs, tools=tools, tokenize=False, add_generation_prompt=False,
    )
    return {"text": text}
```
Applied via `.map()` → consumed by `SFTConfig(dataset_text_field="text")`.

## Data loading patterns
- **Training**: `load_dataset("Nanthasit/sakthai-combined-v7", split="train")`
- **Supplement**: `load_dataset("Nanthasit/sakthai-irrelevance-supplement", split="train")` — concatenated
- **Eval**: `load_dataset("Nanthasit/sakthai-combined-v7", split="test")` or raw JSONL from `sakthai-bench-v2`
- **Subsampling**: CPU script uses `LIMIT=20`; `None` content → `""` (Arrow type consistency)

## Bench-v2 format (eval-only)
- Raw JSONL (list of dicts, no Dataset/Arrow)
- Fields: `messages`, `tools`, `gold_calls`, `category`, `held_out_tool`
- Categories: simple, parallel, irrelevance + sub-categories
- Scores: selection, arguments, strict

## Dataset evolution
```
SimpleToolCalling → sakthai-combined-v6 → sakthai-combined-v7
                                       └→ sakthai-bench-v1 → sakthai-bench-v2
```

## All datasets reference
| Repo | Rows | Downloads | Split | Purpose |
|---|---|---|---|---|
| `sakthai-combined-v7` | 2,424 | 0 | train+test | Latest (en+th). Cross-links still point to v6! |
| `sakthai-combined-v6` | 2,116 | 175 | train+test | Previous version |
| `sakthai-irrelevance-supplement` | 60 | 0 | train | When NOT to call tools |
| `sakthai-bench-v2` | 500 | 0 | test | Official BFCL benchmark |
| `sakthai-bench-v1` | 235 | 0 | test | Early benchmark (superseded) |
| `food-penguin-v1` | 648 | 51 | train | Restaurant domain |
| `SimpleToolCalling` | — | 52 | — | Deprecated, gated |
| `sakthai-kaggle-notebooks` | — | 103 | — | Training notebooks |

## Data considerations
- All datasets stream from HF Hub (no local copies)
- Irrelevance supplement is optional (graceful fallback in training script)
- SimpleToolCalling is gated (requires login agreement)
- Prefer v7 over v6; prefer bench-v2 over bench-v1
