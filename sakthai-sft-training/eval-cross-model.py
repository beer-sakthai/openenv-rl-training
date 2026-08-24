#!/usr/bin/env python3
# /// script
# dependencies = ["torch", "transformers", "accelerate", "peft"]
# ///
"""Cross-model eval: all SakThai merged models on the exact same sakthai-bench-v2 with the exact same scorer.

One script, N models, one benchmark, one scorer. This fixes benchmark inconsistency
(0.5B at 91.2% vs 1.5B at 48.2%) by using identical rendering + scoring everywhere.

- Data: raw test.jsonl from sakthai-bench-v2 (bypasses Hub metadata bug)
- Rendering: tokenizer.apply_chat_template(tools=...) - identical to training rendering
- Scorer: normalized JSON match -> selection / arguments / strict
- LoRA repos (id contains "lora") are auto-loaded via peft on top of their base model.

Env:
  MODELS   comma-separated repo ids (default: 0.5b/1.5b/7b merged context models)
  SAMPLE   random subset size, 0 = all (default 0)
  BATCH    inference batch size (default 4)
  MAX_NEW  max new tokens (default 256)
  DUMP     number of per-model row dumps (default 0)
  OUT      output dir for results yaml (default ".eval_results")
"""
import collections
import datetime
import gc
import json
import os
import random
import re
import sys
import time
from urllib.request import urlopen
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_MODELS = [
    "Nanthasit/sakthai-context-0.5b-merged",
    "Nanthasit/sakthai-context-1.5b-merged",
    "Nanthasit/sakthai-context-7b-merged",
]
MODELS = [m.strip() for m in os.environ.get("MODELS", ",".join(DEFAULT_MODELS)).split(",") if m.strip()]
SAMPLE = int(os.environ.get("SAMPLE", "0"))
BATCH = int(os.environ.get("BATCH", "4"))
MAX_NEW = int(os.environ.get("MAX_NEW", "256"))
DUMP = int(os.environ.get("DUMP", "0"))
OUT = os.environ.get("OUT", ".eval_results")

URL = "https://huggingface.co/datasets/Nanthasit/sakthai-bench-v2/resolve/main/data/test.jsonl"
print(f"Loading {URL} ...", flush=True)
with urlopen(URL) as f:
    TEST = [json.loads(line) for line in f.read().decode().strip().splitlines()]
if SAMPLE:
    random.seed(42)
    TEST = random.sample(TEST, min(SAMPLE, len(TEST)))
N = len(TEST)
print(f"Loaded {N} test rows", flush=True)


def parse_tool_calls(text):
    calls = []
    for m in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            args = obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            calls.append({"name": obj.get("name", ""), "arguments": args})
        except json.JSONDecodeError:
            pass
    return calls


def norm_args(a):
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except json.JSONDecodeError:
            return str(a)
    if isinstance(a, dict):
        return {k: norm_args(v) for k, v in sorted(a.items()) if v is not None}
    return a


def norm_call(c):
    return {"name": c.get("name", ""), "arguments": norm_args(c.get("arguments", {}))}


def match_score(gold_calls, pred_calls):
    gs = {json.dumps(norm_call(c), sort_keys=True) for c in gold_calls}
    ps = {json.dumps(norm_call(c), sort_keys=True) for c in pred_calls}
    if not gs and not ps:
        return True, True
    correct = gs == ps
    args_ok = all(any(g["name"] == p["name"] and g["arguments"] == p["arguments"]
                      for p in pred_calls) for g in gold_calls) if pred_calls else False
    return correct, args_ok


def load_model(repo_id):
    is_lora = "lora" in repo_id.lower() or "adapter" in repo_id.lower()
    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda else torch.float32
    if is_lora:
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(repo_id=repo_id, filename="adapter_config.json")
        with open(config_path, "r") as f:
            base = json.load(f)["base_model_name_or_path"]
        print(f"Loading base {base} + LoRA {repo_id} ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(base)
        base_model = AutoModelForCausalLM.from_pretrained(
            base, dtype=dtype, device_map="auto" if cuda else None, low_cpu_mem_usage=True)
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, repo_id)
    else:
        print(f"Loading {repo_id} ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForCausalLM.from_pretrained(
            repo_id, dtype=dtype, device_map="auto" if cuda else None, low_cpu_mem_usage=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    print("Model loaded", flush=True)
    return tokenizer, model


def pct(score_list):
    return 100.0 * sum(score_list) / len(score_list) if score_list else float("nan")


def summarize(rows):
    out = {}
    for cat in sorted(rows):
        r = rows[cat]
        out[cat] = {
            "n": len(r["sel"]),
            "selection": pct(r["sel"]),
            "arguments": pct(r["args"]),
            "strict": pct(r["strict"]),
        }
    return out


def eval_repo(repo_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 70}\nModel: {repo_id}  device: {device}\n{'=' * 70}", flush=True)
    tokenizer, model = load_model(repo_id)
    results = collections.defaultdict(lambda: {"sel": [], "args": [], "strict": []})
    held_results = collections.defaultdict(lambda: {"sel": [], "args": [], "strict": []})
    t0 = time.time()
    for i in range(0, N, BATCH):
        batch = TEST[i:i + BATCH]
        prompts = []
        for row in batch:
            msgs = list(row.get("messages", []))
            while msgs and msgs[-1].get("role") in ("assistant", "tool"):
                msgs.pop()
            prompts.append(tokenizer.apply_chat_template(
                msgs, tools=row.get("tools") or None,
                tokenize=False, add_generation_prompt=True,
            ))
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for j, row in enumerate(batch):
            gen = tokenizer.decode(outputs[j][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred_calls = parse_tool_calls(gen)
            gold_calls = row.get("gold_calls", [])
            category = row.get("category", "unknown")
            correct, args_ok = match_score(gold_calls, pred_calls)
            target = held_results if row.get("held_out_tool", False) else results
            target[category]["sel"].append(correct)
            target[category]["args"].append(args_ok)
            target[category]["strict"].append(correct and args_ok)
            if DUMP and j < DUMP:
                print(f"--- Row {i + j} ({category}) ---", flush=True)
                print(f"GOLD: {gold_calls}", flush=True)
                print(f"PRED: {pred_calls}", flush=True)
                print(f"RAW: {gen[:200]!r}", flush=True)
        rate = (i + len(batch)) / max(1e-9, time.time() - t0)
        print(f"  [{i + len(batch)}/{N}] {100.0 * (i + len(batch)) / N:.0f}%  {rate:.2f} rows/s  {time.time() - t0:.0f}s", flush=True)

    summary = summarize(results)
    merged_irr = {c: summary[c] for c in ("irrelevance_tools", "irrelevance_no_tools") if c in summary}
    irr_n = sum(merged_irr[c]["n"] for c in merged_irr)
    irr = {m: (sum(merged_irr[c][m] * merged_irr[c]["n"] for c in merged_irr) / irr_n if irr_n else float("nan"))
           for m in ("selection", "arguments", "strict")}
    held = summarize(held_results)
    held_avg = {m: sum(held[c][m] for c in held) / len(held) if held else float("nan")
                for m in ("selection", "arguments", "strict")}

    print(f"\n--- {repo_id} on sakthai-bench-v2 ---", flush=True)
    print(f"  {'category':20s} {'n':>5} {'selection':>9} {'arguments':>9} {'strict':>9}", flush=True)
    for cat in sorted(summary):
        s = summary[cat]
        print(f"  {cat:20s} {s['n']:>5} {s['selection']:>8.1f}% {s['arguments']:>8.1f}% {s['strict']:>8.1f}%", flush=True)
    if irr_n:
        print(f"  {'irrelevance (merged)':20s} {irr_n:>5} {irr['selection']:>8.1f}% {irr['arguments']:>8.1f}% {irr['strict']:>8.1f}%", flush=True)
    overall = {"selection": pct([v for r in results.values() for v in r["sel"]]),
               "arguments": pct([v for r in results.values() for v in r["args"]]),
               "strict": pct([v for r in results.values() for v in r["strict"]])}
    print(f"  {'OVERALL':20s} {N:>5} {overall['selection']:>8.1f}% {overall['arguments']:>8.1f}% {overall['strict']:>8.1f}%", flush=True)
    if held and held_avg["selection"] == held_avg["selection"]:
        h_n = sum(held[c]["n"] for c in held)
        print(f"  {'HELD-OUT':20s} {h_n:>5} {held_avg['selection']:>8.1f}% {held_avg['arguments']:>8.1f}% {held_avg['strict']:>8.1f}%", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"summary": summary, "irrelevance": irr, "overall": overall,
            "held": held, "held_avg": held_avg}


def write_results(repo_id, res):
    import pathlib
    outdir = pathlib.Path(OUT)
    outdir.mkdir(parents=True, exist_ok=True)
    safe = repo_id.split("/")[-1]
    ov = res["overall"]
    badge = outdir / f"{safe}.sakthai-bench-v2.yaml"
    badge.write_text(
        f"task:\n  - text-generation\n"
        f"dataset:\n  - sakthai-bench-v2\n"
        f"metrics:\n"
        f"  - selection: {ov['selection']:.1f}\n"
        f"  - arguments: {ov['arguments']:.1f}\n"
        f"  - strict: {ov['strict']:.1f}\n",
        encoding="utf-8")
    detail = outdir / f"{safe}.cross-model.yaml"
    detail.write_text(json.dumps({
        "model": repo_id,
        "benchmark": "sakthai-bench-v2",
        "n_rows": N,
        "date": datetime.date.today().isoformat(),
        "overall": ov,
        "per_category": res["summary"],
        "held_out": res["held_avg"],
    }, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {badge}", flush=True)
    print(f"Wrote {detail}", flush=True)


def main():
    all_res = {}
    for repo_id in MODELS:
        all_res[repo_id] = eval_repo(repo_id)
        write_results(repo_id, all_res[repo_id])

    print("\n" + "=" * 78, flush=True)
    print("UNIFIED CROSS-MODEL RESULTS - sakthai-bench-v2 (same data, same scorer)", flush=True)
    print("=" * 78, flush=True)
    cats = ["simple", "parallel", "irrelevance", "overall"]
    header = f"  {'model':<36s}" + "".join(f"{c:>12s}" for c in ["simple", "parallel", "irrev", "OVERALL"])
    print(header, flush=True)
    print("  " + "-" * 74, flush=True)
    for repo_id, res in all_res.items():
        label = repo_id.replace("Nanthasit/", "")
        merged_irr = res["irrelevance"]
        cells = []
        for cat in cats:
            if cat == "irrelevance":
                m = merged_irr["selection"]
            elif cat == "overall":
                m = res["overall"]["selection"]
            else:
                m = res["summary"][cat]["selection"] if cat in res["summary"] else float("nan")
            cells.append(m)
        row = f"  {label:<36s}" + "".join(f"{c:>10.1f}%" if c == c else f"{'-':>10}" for c in cells)
        print(row, flush=True)
    print("  " + "-" * 74, flush=True)
    print("  selection accuracy per category", flush=True)

    print("\n" + "=" * 78, flush=True)
    print("STRICT ACCURACY (selection AND arguments) per category", flush=True)
    print("=" * 78, flush=True)
    print(header, flush=True)
    print("  " + "-" * 74, flush=True)
    for repo_id, res in all_res.items():
        label = repo_id.replace("Nanthasit/", "")
        merged_irr = res["irrelevance"]
        cells = []
        for cat in cats:
            if cat == "irrelevance":
                m = merged_irr["strict"]
            elif cat == "overall":
                m = res["overall"]["strict"]
            else:
                m = res["summary"][cat]["strict"] if cat in res["summary"] else float("nan")
            cells.append(m)
        row = f"  {label:<36s}" + "".join(f"{c:>10.1f}%" if c == c else f"{'-':>10}" for c in cells)
        print(row, flush=True)
    print("  " + "-" * 74, flush=True)

    print("\nDone. Badge yamls in ./%s/ - push each to its repo as .eval_results/sakthai-bench-v2.yaml" % OUT, flush=True)


if __name__ == "__main__":
    main()
