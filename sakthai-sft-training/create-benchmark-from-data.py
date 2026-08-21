#!/usr/bin/env python3
"""
Create a proper BFCL-style benchmark from v8 cycle data.
Different from all previous: this creates an EVAL BENCHMARK (not training data).
Uses the exact eval_bench.py scorer format from sakthai-bench-v2.

Categories:
- simple: 1 tool call expected
- parallel: 2+ tool calls expected  
- irrelevance_tools: tools offered, model must not call
- irrelevance_no_tools: no tools, model must not call
"""
import json, random, glob, hashlib
from pathlib import Path

random.seed(42)
OUT = Path("v8-benchmark")
OUT.mkdir(exist_ok=True)

# Load all v8 examples
v8_data = []
for f in sorted(glob.glob("cycle-100-v8/iter-*.jsonl")):
    with open(f, encoding="utf-8") as fh:
        ex = json.loads(fh.read())
        v8_data.append(ex)

print(f"Loaded {len(v8_data)} v8 examples")

# Convert to bench-v2 format
bench_rows = []
categories_used = {"simple": 0, "parallel": 0, "irrelevance_tools": 0, "irrelevance_no_tools": 0}

for ex in v8_data:
    msgs = ex.get("messages", [])
    tools = ex.get("tools", [])
    
    # Find assistant turn
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant":
            gold_calls = []
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except: args = {}
                gold_calls.append({"name": name, "arguments": args})
            
            # Determine category
            if not gold_calls and tools:
                cat = "irrelevance_tools"
            elif not gold_calls and not tools:
                cat = "irrelevance_no_tools"
            elif len(gold_calls) == 1:
                cat = "simple"
            else:
                cat = "parallel"
            
            # Create bench row
            row = {
                "messages": msgs[:i+1],
                "tools": tools,
                "gold_calls": gold_calls,
                "category": cat,
                "held_out_tool": False,
                "multi_turn": any(m.get("role") == "tool" for m in msgs[:i]),
                "in_v1": False,
            }
            bench_rows.append(row)
            categories_used[cat] = categories_used.get(cat, 0) + 1
            break

print(f"Created {len(bench_rows)} benchmark rows")
for cat, count in categories_used.items():
    print(f"  {cat}: {count}")

# Write as JSONL (exact bench-v2 format)
bench_path = OUT / "bench.jsonl"
with open(bench_path, "w", encoding="utf-8") as f:
    for row in bench_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"\nWrote {bench_path}")

# Write categorized summary
summary = {
    "name": "v8-cycle-benchmark",
    "description": "BFCL-style benchmark derived from cycle-100-v8 data",
    "total_rows": len(bench_rows),
    "categories": categories_used,
    "source": "cycle-100-v8",
    "evaluation_script": "eval_bench.py",
}
summary_path = OUT / "summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Wrote {summary_path}")

# Create a copy of eval_bench.py adapted for this benchmark
eval_script = """#!/usr/bin/env python3
# Auto-generated eval script for v8-cycle-benchmark
# Usage: SAK_MODELS=model_id uv run python eval_v8_bench.py
import os, json, re, time, collections
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BENCH = os.path.dirname(os.path.abspath(__file__))
MODELS = [m.strip() for m in os.environ.get("SAK_MODELS", "").split(",") if m.strip()]
BATCH = int(os.environ.get("SAK_BATCH", "8"))

with open(os.path.join(BENCH, "bench.jsonl")) as f:
    ROWS = [json.loads(line) for line in f]
print(f"Loaded {len(ROWS)} benchmark rows")

_TC = re.compile(r"<tool_call>\\s*(\\{.*?\\})\\s*</tool_call>", re.DOTALL)
def norm(v):
    if isinstance(v, str):
        s = v.strip()
        try: return norm(json.loads(s))
        except: return s.lower()
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, dict): return {k: norm(x) for k, x in sorted(v.items())}
    if isinstance(v, list): return [norm(x) for x in v]
    return v

def evaluate(repo_id):
    tok = AutoTokenizer.from_pretrained(repo_id)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map="auto")
    m.eval()
    
    results = collections.defaultdict(lambda: [0, 0])
    for row in ROWS:
        cat, gold = row["category"], row["gold_calls"]
        gold_names = [c["name"] for c in gold]
        prompt = ""
        for msg in row["messages"]:
            if msg["role"] == "user": prompt += f"<|im_start|>user\\n{msg['content']}<|im_end|>\\n"
            if msg["role"] == "assistant" and not msg.get("tool_calls"):
                prompt += f"<|im_start|>assistant\\n{msg['content']}<|im_end|>\\n"
        prompt += "<|im_start|>assistant\\n"
        
        inputs = tok(prompt, return_tensors="pt").to(m.device)
        out = m.generate(**inputs, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        pred = []
        for mm in _TC.findall(gen):
            try:
                d = json.loads(mm)
                a = d.get("arguments", {})
                if isinstance(a, str):
                    try: a = json.loads(a)
                    except: a = {}
                pred.append({"name": d.get("name",""), "arguments": a})
            except: pass
        
        pred_names = [c["name"] for c in pred]
        ok = False
        if cat.startswith("irrelevance"):
            ok = len(pred_names) == 0
        elif cat == "simple":
            ok = bool(gold_names) and gold_names[0] in pred_names
        else:
            ok = not (collections.Counter(gold_names) - collections.Counter(pred_names))
        
        results[cat][0] += int(ok)
        results[cat][1] += 1
    
    print(f"\\n=== {repo_id} ===")
    tc = tt = 0
    for c in ("simple", "parallel", "irrelevance_tools", "irrelevance_no_tools"):
        p, t = results[c]
        tc += p; tt += t
        print(f"  {c:25s} {p:3d}/{t:3d} = {100*p/t:.1f}%" if t else f"  {c:25s} n/a")
    print(f"  {'OVERALL':25s} {tc:3d}/{tt:3d} = {100*tc/tt:.1f}%")

for repo in MODELS:
    evaluate(repo)
"""

eval_path = OUT / "eval_v8_bench.py"
with open(eval_path, "w") as f:
    f.write(eval_script)
print(f"Wrote {eval_path}")

print(f"\nTo run: SAK_MODELS=Nanthasit/sakthai-context-1.5b-merged uv run python {eval_path}")
print(f"Benchmark ready: {OUT.resolve()}/")
