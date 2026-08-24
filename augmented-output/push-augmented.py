"""
Push augmented dataset to Hugging Face Hub.
Usage: uv run python push-augmented.py
Requires HF_TOKEN env var.
"""
import os, json
from datasets import Dataset
from huggingface_hub import HfApi

rows = []
with open("augmented-output/all-augmented.jsonl") as f:
    for line in f:
        rows.append(json.loads(line))

ds = Dataset.from_list(rows)
print(f"Created dataset: {len(rows)} rows, features: {ds.features}")

# Push as new version (v8) or update v7
# Option A: Push as v8
ds.push_to_hub("Nanthasit/sakthai-combined-v8", split="train")
print("✅ Pushed as Nanthasit/sakthai-combined-v8")

# Option B: Append to v7 (uncomment to use)
# from datasets import load_dataset, concatenate_datasets
# existing = load_dataset("Nanthasit/sakthai-combined-v7", split="train")
# combined = concatenate_datasets([existing, ds])
# combined.push_to_hub("Nanthasit/sakthai-combined-v7", split="train")
