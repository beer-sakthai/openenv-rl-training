# HF Hub Improvements — Gap Analysis

Audit date: 2026-07-30
Checked: 13 models, 8 datasets, 3 spaces

## Gaps Found

| # | Resource | Downloads | Gap | Impact |
|---|---|---|---|---|
| 1 | `sakthai-context-7b-merged` | 744 | **No GGUF** | 1.5B GGUF = 1,599 dl (#1). 7B GGUF would be #2. |
| 2 | `sakthai-context-7b-tools` | 399 | No eval results | Missing benchmark badges on model page |
| 3 | `sakthai-context-1.5b-tools` | 349 | No eval results | Missing benchmark badges on model page |
| 4 | `sakthai-context-7b-128k` | 506 | No inference provider | 506 visitors can't try it in browser |

## Fixes

### 1. Create 7B GGUF

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release

huggingface-cli download Nanthasit/sakthai-context-7b-merged --local-dir ./sakthai-7b

python convert_hf_to_gguf.py ./sakthai-7b --outfile ./sakthai-7b-fp16.gguf

./build/bin/quantize ./sakthai-7b-fp16.gguf ./sakthai-7b-q4_k_m.gguf q4_K_M

huggingface-cli upload Nanthasit/sakthai-context-7b-merged ./sakthai-7b-q4_k_m.gguf .
```

Expected impact: 500+ additional downloads within first month.

### 2. Add .eval_results/ for 7b-tools and 1.5b-tools

```python
from huggingface_hub import HfApi
api = HfApi()

for model, score in [("Nanthasit/sakthai-context-7b-tools", 57.0),
                     ("Nanthasit/sakthai-context-1.5b-tools", 48.2)]:
    yaml = f"""task:
  - text-generation
dataset:
  - sakthai-bench-v2
metrics:
  - selection: {score}
    name: Selection Accuracy
    verified: false
"""
    api.upload_file(
        path_or_fileobj=yaml.encode(),
        path_in_repo=".eval_results/sakthai-bench-v2.yaml",
        repo_id=model
    )
```

This enables benchmark badges on model cards.

### 3. Request inference provider for 7b-128k

No automated way. Manual steps:
1. Visit https://huggingface.co/Nanthasit/sakthai-context-7b-128k
2. Click "Ask for provider support" near the inference widget area
3. Alternative: Contact Featherless AI directly (they host 1.5B already)
