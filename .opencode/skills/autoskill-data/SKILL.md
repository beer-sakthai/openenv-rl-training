---
name: autoskill-data
description: AUTO-TRIGGER on dataset operations — inspect v7/v8, count rows, validate format, fix Arrow schema issues, combine datasets, push new versions. Runs every time you mention dataset, JSONL, data quality, or augmentation. Phase: DATA (universal).
---

# Autoskill: Dataset Operations

Auto-triggers on: "dataset", "JSONL", "data quality", "v7", "v8", "load_dataset", "push dataset", "combine datasets"

## Quick diagnostics

### Count rows
```python
from datasets import load_dataset
for v in ["7", "8"]:
    ds = load_dataset(f"Nanthasit/sakthai-combined-v{v}", split="train")
    print(f"v{v}: {len(ds)} rows")
```

### Fix None -> [] for Arrow
```python
for row in data:
    for m in row.get("messages", []):
        if m.get("content") is None: m["content"] = ""
        if "tool_calls" in m and m["tool_calls"] is None: m["tool_calls"] = []
```

### Combine v7 + v8
```python
v7 = load_dataset("Nanthasit/sakthai-combined-v7", split="train")
v8 = load_dataset("Nanthasit/sakthai-combined-v8", split="train")
combined = concatenate_datasets([v7, v8])
print(f"Combined: {len(combined)} rows")
```
