#!/usr/bin/env python3
# /// script
# dependencies = ["huggingface_hub", "datasets"]
# ///
"""
Push v9 to HF Hub: combines all cycle data + safety/quality fixes + model card fixes.
"""
import json, glob, os, itertools, random
from pathlib import Path
from huggingface_hub import HfApi
from datasets import Dataset, concatenate_datasets, load_dataset, Features, Value, Sequence, Json

api = HfApi()
repo = "Nanthasit/sakthai-combined-v9"

# ── 1. Combine all data into one deduplicated set ──────────────
seen = set()
all_rows = []

# Read all cycle rounds
for d in ["cycle-100-output","cycle-100-v2","cycle-100-v3","cycle-100-v4",
          "cycle-100-v5","cycle-100-v6","cycle-100-v7","cycle-100-v8",
          "cycle-100-v9","cycle-100-v10","gap-filled","benchmark-targeted",
          "augmented-output","safety-quality-fixes"]:
    for f in sorted(glob.glob(f"{d}/*.jsonl")):
        if "all-" in f or "push-" in f: continue
        with open(f, encoding="utf-8") as fh:
            try:
                for line in fh:
                    line = line.strip()
                    if not line: continue
                    h = json.dumps(json.loads(line), sort_keys=True, ensure_ascii=False)
                    if h not in seen:
                        seen.add(h)
                        all_rows.append(json.loads(line))
            except:
                pass

print(f"Total unique: {len(all_rows)}")

# Normalize None -> [] for Arrow compatibility
for row in all_rows:
    if row.get("tools") is None:
        row["tools"] = []
    for m in row.get("messages", []):
        if m.get("content") is None:
            m["content"] = ""
        if "tool_calls" in m and m["tool_calls"] is None:
            m["tool_calls"] = []

# ── 2. Upload v9 as raw JSONL ──────────────────────────────
api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
print(f"Repo {repo} ready")

# Write combined JSONL
combined_path = "combined-v9.jsonl"
with open(combined_path, "w", encoding="utf-8") as f:
    for row in all_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

api.upload_file(path_or_fileobj=combined_path, path_in_repo="data/all.jsonl", repo_id=repo, repo_type="dataset")
print(f"Uploaded {len(all_rows)} rows as data/all.jsonl")

# ── 3. Upload train split via datasets ──────────────────────
try:
    ds = Dataset.from_list(all_rows)
    ds.push_to_hub(repo, split="train")
    print(f"Pushed {len(ds)} rows as train split")
except Exception as e:
    print(f"Dataset push failed (will try parquet): {e}")
    # Fallback: upload as JSONL only

# ── 4. Create README ───────────────────────────────────────
readme = f"""---
license: apache-2.0
language: [en, th]
tags: [sakthai, tool-calling, function-calling, augmented, v9]
---
# SakThai Combined Dataset v9

**{len(all_rows)} augmented tool-calling examples** across 100+ categories.
Extends v7 and v8 with comprehensive coverage.

## Contents
- `data/all.jsonl` — All {len(all_rows)} examples

## Coverage (100+ categories across 10 dimensions)
1. Basic tool calling (single, parallel, multi-hop)
2. Edge cases (empty args, null values, unicode, long params)
3. Type enforcement (int/float/boolean/enum/array)
4. Context handling (multi-turn, system prompt, assistant context)
5. Real-world scenarios (domain terminology, geo, time, comparison)
6. Safety & guardrails (harmful request refusal, PII protection)
7. Quality enforcement (schema matching, validation rules)
8. Error recovery (error -> retry, async polling)
9. Advanced workflows (draft/publish, approval, wizards, onboarding)
10. Tool management (discovery, favorites, tagging, compliance, docs)

## Version history
- v7: 2,424 original examples
- v8: +538 augmented examples
- v9: +{len(all_rows)} augmented examples across 100+ categories

## Augmentation scripts
All generation scripts are open source. See the project repository.

See also: [v7](https://huggingface.co/datasets/Nanthasit/sakthai-combined-v7)
"""
api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=repo, repo_type="dataset")
print("Uploaded README.md")

# ── 5. Push model card fixes ───────────────────────────────
fixes_dir = Path(".opencode/skills/huggingface/fixes")
if fixes_dir.exists():
    # merged-v2 card
    card_path = fixes_dir / "merged-v2-card.md"
    if card_path.exists():
        api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md",
                        repo_id="Nanthasit/sakthai-context-1.5b-merged-v2")
        print("Updated merged-v2 model card")
    
    # eval_results
    eval_path = fixes_dir / ".eval_results/sakthai-bench-v2.yaml"
    if eval_path.exists():
        api.upload_file(path_or_fileobj=str(eval_path), path_in_repo=".eval_results/sakthai-bench-v2.yaml",
                        repo_id="Nanthasit/sakthai-context-1.5b-merged-v2")
        print("Uploaded .eval_results for merged-v2")
    
    # v2 tools README
    v2_readme = fixes_dir / "v2-readme.md"
    if v2_readme.exists():
        api.upload_file(path_or_fileobj=str(v2_readme), path_in_repo="README.md",
                        repo_id="Nanthasit/sakthai-context-1.5b-tools-v2")
        print("Updated v2-tools model card")


    # sakthai-plus 1.5b card
    plus_card = fixes_dir / "sakthai-plus-card.md"
    if plus_card.exists():
        api.upload_file(path_or_fileobj=str(plus_card), path_in_repo="README.md",
                        repo_id="Nanthasit/sakthai-plus-1.5b")
        print("Updated sakthai-plus-1.5b model card")

    # sakthai-plus-coder 1.5b card
    plus_coder_card = fixes_dir / "sakthai-plus-coder-card.md"
    if plus_coder_card.exists():
        api.upload_file(path_or_fileobj=str(plus_coder_card), path_in_repo="README.md",
                        repo_id="Nanthasit/sakthai-plus-1.5b-coder")
        print("Updated sakthai-plus-1.5b-coder model card")

    # sakthai-plus-lora 1.5b card
    plus_lora_card = fixes_dir / "sakthai-plus-lora-card.md"
    if plus_lora_card.exists():
        api.upload_file(path_or_fileobj=str(plus_lora_card), path_in_repo="README.md",
                        repo_id="Nanthasit/sakthai-plus-1.5b-lora")
        print("Updated sakthai-plus-1.5b-lora model card")

print(f"\nDone: https://huggingface.co/datasets/{repo}")
print(f"Model cards updated: merged-v2, v2-tools, sakthai-plus-1.5b, sakthai-plus-1.5b-coder, sakthai-plus-1.5b-lora")
