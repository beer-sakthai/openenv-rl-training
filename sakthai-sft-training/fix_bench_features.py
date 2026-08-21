"""Fix: push a clean version of sakthai-bench-v2 with consistent types."""

import json, os, sys, urllib.request

REPO = "Nanthasit/sakthai-bench-v2"
JSONL_URL = "https://huggingface.co/datasets/" + REPO + "/resolve/main/data/test.jsonl"

print(f"Fetching {JSONL_URL} ...")
with urllib.request.urlopen(JSONL_URL) as f:
    rows = [json.loads(line) for line in f.read().decode().strip().splitlines()]
print(f"Loaded {len(rows)} rows")

# Normalize: None content -> "" (Arrow can't mix str and NoneType)
for row in rows:
    for msg in row.get("messages", []):
        if msg.get("content") is None:
            msg["content"] = ""
        if "tool_calls" in msg and msg["tool_calls"] is None:
            msg["tool_calls"] = []

from datasets import Dataset
ds = Dataset.from_list(rows)
print(f"Created Dataset: {len(ds)} rows, features: {ds.features}")

ds.push_to_hub(REPO, split="test")
print("Done. Clean dataset pushed.")
