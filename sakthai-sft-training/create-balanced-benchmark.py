#!/usr/bin/env python3
"""
Create a balanced BFCL-style benchmark from ALL cycle data.
Different from all previous work: this is a PROPER EVAL BENCHMARK (not training data).
Balanced across simple/parallel/irrelevance categories, in exact bench-v2 format.
"""
import json, glob, random, collections
from pathlib import Path
from huggingface_hub import HfApi

random.seed(42)
TARGET_PER_CAT = 50  # 50 per category = 200 total

# Load all cycle data
all_examples = []
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
                    all_examples.append(json.loads(line))
            except:
                pass

print(f"Total loaded: {len(all_examples)}")

# Classify and categorize
by_cat = {"simple": [], "parallel": [], "irrelevance_tools": [], "irrelevance_no_tools": []}

for ex in all_examples:
    msgs = ex.get("messages", [])
    tools = ex.get("tools", [])
    
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant":
            gold_calls = []
            for tc_raw in (m.get("tool_calls") or []):
                if isinstance(tc_raw, str):
                    try: tc_raw = json.loads(tc_raw)
                    except: continue
                if not isinstance(tc_raw, dict): continue
                fn = tc_raw.get("function", {})
                if isinstance(fn, str):
                    try: fn = json.loads(fn)
                    except: fn = {}
                if not isinstance(fn, dict): continue
                name = fn.get("name", "")
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except: args = {}
                gold_calls.append({"name": name, "arguments": args})
            
            nc = len(gold_calls)
            if nc == 0 and tools: cat = "irrelevance_tools"
            elif nc == 0 and not tools: cat = "irrelevance_no_tools"
            elif nc == 1: cat = "simple"
            else: cat = "parallel"
            
            row = {
                "messages": msgs[:i+1],
                "tools": tools,
                "gold_calls": gold_calls,
                "category": cat,
                "held_out_tool": False,
                "multi_turn": any(m2.get("role") == "tool" for m2 in msgs[:i]),
                "in_v1": False,
            }
            by_cat[cat].append(row)
            break

for cat, rows in by_cat.items():
    print(f"  {cat:25s}: {len(rows)} available")

# Sample balanced set
bench_rows = []
for cat in ["simple", "parallel", "irrelevance_tools", "irrelevance_no_tools"]:
    pool = by_cat.get(cat, [])
    random.shuffle(pool)
    selected = pool[:min(TARGET_PER_CAT, len(pool))]
    bench_rows.extend(selected)
    print(f"  Sampled {cat}: {len(selected)}")

random.shuffle(bench_rows)
print(f"\nTotal benchmark rows: {len(bench_rows)}")

# Write output
OUT = Path("sakthai-cycle-bench")
OUT.mkdir(exist_ok=True)

bench_path = OUT / "data/test.jsonl"
with open(bench_path, "w", encoding="utf-8") as f:
    for row in bench_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# Summary
summary_path = OUT / "summary.json"
cats = collections.Counter(r["category"] for r in bench_rows)
with open(summary_path, "w") as f:
    json.dump({"total": len(bench_rows), "categories": dict(cats),
               "multi_turn": sum(1 for r in bench_rows if r["multi_turn"])}, f, indent=2)

# Push to HF Hub
api = HfApi()
repo = "Nanthasit/sakthai-cycle-bench"
api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)

api.upload_file(path_or_fileobj=str(bench_path), path_in_repo="data/test.jsonl",
                repo_id=repo, repo_type="dataset")
api.upload_file(path_or_fileobj=str(summary_path), path_in_repo="summary.json",
                repo_id=repo, repo_type="dataset")

readme = f"""---
license: apache-2.0
tags: [sakthai, benchmark, tool-calling, function-calling]
---
# SakThai Cycle Benchmark

**{len(bench_rows)} balanced BFCL-style benchmark rows** derived from 10 cycle rounds.

| Category | Count |
|---|---|
| simple | {cats.get('simple',0)} |
| parallel | {cats.get('parallel',0)} |
| irrelevance_tools | {cats.get('irrelevance_tools',0)} |
| irrelevance_no_tools | {cats.get('irrelevance_no_tools',0)} |
| **Total** | **{len(bench_rows)}** |

Multi-turn: {sum(1 for r in bench_rows if r['multi_turn'])} rows

Use with eval_bench.py:
```
SAK_MODELS=Nanthasit/sakthai-context-1.5b-merged \\
SAK_BENCH=Nanthasit/sakthai-cycle-bench \\
uv run python eval_bench.py
```
"""
api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                repo_id=repo, repo_type="dataset")

print(f"\nPushed: https://huggingface.co/datasets/{repo}")
print(f"Run: SAK_MODELS=model_id SAK_BENCH={repo} uv run python eval_bench.py")
