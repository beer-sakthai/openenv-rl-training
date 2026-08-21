#!/usr/bin/env python3
# /// script
# dependencies = ["datasets", "huggingface_hub"]
# ///
"""Push all augmented datasets to Hugging Face Hub as sakthai-combined-v8.
Uploads as raw JSONL files (avoids pyarrow schema issues with nested tool_calls)."""
import json, os
from huggingface_hub import HfApi

api = HfApi()
repo = "Nanthasit/sakthai-combined-v8"

# 1. Create repo
api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)

# 2. Upload training-data-ready.jsonl (538 augmented examples)
api.upload_file(
    path_or_fileobj="training-data-ready.jsonl",
    path_in_repo="data/augmented.jsonl",
    repo_id=repo,
    repo_type="dataset",
)
print("Uploaded augmented.jsonl")

# 3. Also upload as train split
api.upload_file(
    path_or_fileobj="training-data-ready.jsonl",
    path_in_repo="data/train.jsonl",
    repo_id=repo,
    repo_type="dataset",
)
print("Uploaded train.jsonl")

# 4. Create a simple README
readme = f"""---
license: apache-2.0
language: [en, th]
tags: [sakthai, tool-calling, function-calling, augmented, v8]
---
# SakThai Combined Dataset v8

{len(open('training-data-ready.jsonl').readlines())} augmented tool-calling examples.
Extends v7 with targeted data addressing benchmark gaps.

## Contents
- `data/augmented.jsonl` — 538 deduplicated augmented examples
- `data/train.jsonl` — same data as train split

## Augmentation strategies
1. Arguments normalization (norm() whitespace/case)
2. Parallel calls (Counter multiset containment)
3. Irrelevance (empty pred_names)
4. Hard negatives (selection accuracy)
5. Held-out tool generalization
6. Argument type coercion
7. Multi-turn context tracking
8. Multi-hop chains
9. Pure tool-calling (no chat dilution)
10. Strict accuracy (selection + arguments)

See also: [v7](https://huggingface.co/datasets/Nanthasit/sakthai-combined-v7)
"""
api.upload_file(
    path_or_fileobj=readme.encode(),
    path_in_repo="README.md",
    repo_id=repo,
    repo_type="dataset",
)
print("Uploaded README.md")
print(f"\nDone: https://huggingface.co/datasets/{repo}")
print(f"   Download: load_dataset('{repo}', split='train')")
