#!/usr/bin/env python3
# /// script
# dependencies = ["transformers>=4.44", "torch"]
# ///
"""Quick eval on cycle-bench. Usage: uv run python run_eval.py [model_id] [sample_size]"""
import json, sys, re, collections, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = sys.argv[1] if len(sys.argv) > 1 else "Nanthasit/sakthai-context-0.5b-merged"
sample = int(sys.argv[2]) if len(sys.argv) > 2 else None

with open("data/test.jsonl") as f:
    rows = [json.loads(line) for line in f]
if sample: rows = rows[:sample]
print(f"Loading {model_id} on CPU, {len(rows)} rows...")

tok = AutoTokenizer.from_pretrained(model_id)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
m.eval()

_TC = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

def norm(v):
    if isinstance(v, str):
        s = v.strip()
        try: return norm(json.loads(s))
        except: return s.lower()
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, dict): return {k: norm(x) for k, x in sorted(v.items())}
    if isinstance(v, list): return [norm(x) for x in v]
    return v

results = collections.defaultdict(lambda: [0, 0])
t0 = time.time()

for idx, row in enumerate(rows):
    cat, gold = row["category"], row["gold_calls"]
    gold_names = [c["name"] for c in gold]
    
    # Build prompt matching the training format
    prompt = ""
    for msg in row["messages"]:
        r = msg["role"]
        c = msg.get("content", "") or ""
        if r == "system": prompt += f"<|im_start|>system\n{c}<|im_end|>\n"
        elif r == "user": prompt += f"<|im_start|>user\n{c}<|im_end|>\n"
        elif r == "assistant" and not msg.get("tool_calls"):
            prompt += f"<|im_start|>assistant\n{c}<|im_end|>\n"
        elif r == "assistant" and msg.get("tool_calls"):
            prompt += f"<|im_start|>assistant\n{c}<|im_end|>\n"
        elif r == "tool": prompt += f"<|im_start|>user\n<tool_response>\n{c}\n</tool_response><|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
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
    
    if cat.startswith("irrelevance"):
        ok = len(pred_names) == 0
    elif cat == "simple":
        ok = bool(gold_names) and gold_names[0] in pred_names
    else:
        ok = not (collections.Counter(gold_names) - collections.Counter(pred_names))
    
    results[cat][0] += int(ok)
    results[cat][1] += 1
    
    if (idx+1) % 10 == 0:
        elapsed = time.time() - t0
        print(f"  [{idx+1}/{len(rows)}] {elapsed:.0f}s elapsed")

print(f"\n=== {model_id} ({time.time()-t0:.0f}s) ===")
tc = tt = 0
for c in ["simple", "parallel", "irrelevance_tools", "irrelevance_no_tools"]:
    p, t = results[c]
    tc += p; tt += t
    pct = f"{100*p/t:.1f}%" if t else "N/A"
    print(f"  {c:25s} {p:3d}/{t:3d} = {pct}")
print(f"  {'OVERALL':25s} {tc:3d}/{tt:3d} = {100*tc/tt:.1f}%")
