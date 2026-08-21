#!/usr/bin/env python3
"""BFCL-style evaluation for SakThai 1.5B models on held-out test split."""
import re, json, gc, collections
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_USER = "Nanthasit"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

_TC = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
TEST = load_dataset(f"{HF_USER}/sakthai-combined-v7", split="test")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def _gold(msg):
    o = [(tc.get("function") or {}).get("name") for tc in (msg.get("tool_calls") or [])]
    o = [n for n in o if n]
    if not o:
        for mm in _TC.findall(msg.get("content") or ""):
            try:
                o.append(json.loads(mm).get("name"))
            except Exception:
                pass
    return [n for n in o if n]

def _pred(t):
    o = []
    for mm in _TC.findall(t):
        try:
            o.append(json.loads(mm).get("name"))
        except Exception:
            pass
    return [n for n in o if n]

def eval_repo(repo_id, label):
    print(f"\nLoading {label}: {repo_id}")
    m = AutoModelForCausalLM.from_pretrained(
        repo_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    m.eval()
    b = collections.defaultdict(lambda: [0, 0])
    for ex in TEST:
        msgs, tools = ex["messages"], (ex.get("tools") or None)
        idx = next((i for i, mm in enumerate(msgs) if mm.get("role") == "assistant"), None)
        if not idx:
            continue
        g = _gold(msgs[idx])
        cat = "irrelevance" if not g else ("simple" if len(g) == 1 else "parallel")
        prompt = tokenizer.apply_chat_template(
            msgs[:idx], tools=tools, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        ).to(m.device)
        with torch.no_grad():
            out = m.generate(
                prompt, max_new_tokens=200, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        pred = _pred(tokenizer.decode(out[0][prompt.shape[1]:], skip_special_tokens=True))
        if cat == "irrelevance":
            ok = len(pred) == 0
        elif cat == "simple":
            ok = g[0] in pred
        else:
            ok = set(g).issubset(set(pred))
        b[cat][0] += int(ok)
        b[cat][1] += 1

    print(f"\n=== BFCL-style: {label} ({repo_id}) ===")
    print(f"{'category':<14}{'pass':>5}{'total':>6}{'acc':>8}")
    tc = tt = 0
    for c in ("simple", "parallel", "irrelevance"):
        p, t = b[c]
        tc += p
        tt += t
        acc = f"{100*p/t:5.1f}%" if t else "  n/a"
        print(f"{c:<14}{p:>5}{t:>6}{acc:>8}")
    overall = f"{100*tc/tt:5.1f}%" if tt else "  n/a"
    print(f"{'OVERALL':<14}{tc:>5}{tt:>6}{overall:>8}")
    del m
    gc.collect()
    torch.cuda.empty_cache()

eval_repo(f"{HF_USER}/sakthai-context-1.5b-merged", "BEFORE (current)")
eval_repo(f"{HF_USER}/sakthai-context-1.5b-merged-v2", "AFTER (v2 new)")
print("\nDone.")
