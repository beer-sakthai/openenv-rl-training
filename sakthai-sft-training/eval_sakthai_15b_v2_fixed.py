"""Evaluate sakthai-context-1.5b-merged-v2 on sakthai-bench-v2.

Loads raw test.jsonl directly (bypasses Hub metadata bug).
No Dataset/Arrow — works on raw list of dicts.
"""
import os, json, re, gc, time, collections, urllib.request, sys
from collections import Counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Nanthasit/sakthai-context-1.5b-merged-v2"
BATCH = 4  # small for CPU
DUMP = 0

# ── Load raw JSONL (list of dicts, no Dataset) ────────────────
URL = "https://huggingface.co/datasets/Nanthasit/sakthai-bench-v2/resolve/main/data/test.jsonl"
print(f"Loading {URL} ...")
with urllib.request.urlopen(URL) as f:
    TEST = [json.loads(line) for line in f.read().decode().strip().splitlines()]
print(f"Loaded {len(TEST)} test rows")

# ── Renderer ──────────────────────────────────────────────────
def _text(c):
    return "" if c is None else (c if isinstance(c, str) else json.dumps(c, ensure_ascii=False))

def _tools_block(tools):
    if not tools: return ""
    sigs = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
    return ("\n\n# Tools\n\nYou may call one or more functions. Signatures are within "
            "<tools></tools>:\n<tools>\n" + sigs + "\n</tools>\n\nFor each call return:\n"
            "<tool_call>\n{\"name\": <name>, \"arguments\": <json>}\n</tool_call>")

def _assistant_body(m):
    body = _text(m.get("content"))
    for tc in (m.get("tool_calls") or []):
        fn = tc.get("function", tc); a = fn.get("arguments", "{}")
        if not isinstance(a, str): a = json.dumps(a, ensure_ascii=False)
        body += ("\n" if body else "") + '<tool_call>\n{"name": "%s", "arguments": %s}\n</tool_call>' % (fn.get("name", ""), a)
    return body

def _render_msg(m, tools_sys):
    r = m.get("role")
    if r == "system": return "<|im_start|>system\n" + _text(m.get("content")) + _tools_block(tools_sys) + "<|im_end|>\n"
    if r == "user":   return "<|im_start|>user\n" + _text(m.get("content")) + "<|im_end|>\n"
    if r == "tool":   return "<|im_start|>user\n<tool_response>\n" + _text(m.get("content")) + "\n</tool_response><|im_end|>\n"
    if r == "assistant": return "<|im_start|>assistant\n" + _assistant_body(m) + "<|im_end|>\n"
    return ""

def render_prompt(row):
    tools = row.get("tools", [])
    msgs = row.get("messages", [])
    prompt = ""
    for i, m in enumerate(msgs):
        prompt += _render_msg(m, tools if i == 0 else [])
    prompt += "<|im_start|>assistant\n"
    return prompt

# ── Scorer ────────────────────────────────────────────────────
def parse_tool_calls(text):
    calls = []
    for m in re.finditer(r'<tool_call>\s*\{(.*?)\}\s*</tool_call>', text, re.DOTALL):
        try:
            obj = json.loads("{" + m.group(1) + "}")
            calls.append({"name": obj.get("name", ""), "arguments": obj.get("arguments", {})})
        except json.JSONDecodeError:
            pass
    return calls

def norm_args(a):
    if isinstance(a, str):
        try: a = json.loads(a)
        except json.JSONDecodeError: return str(a)
    if isinstance(a, dict):
        return {k: norm_args(v) for k, v in sorted(a.items()) if v is not None}
    return a

def norm_call(c):
    return {"name": c.get("name", ""), "arguments": norm_args(c.get("arguments", {}))}

def match_score(gold_calls, pred_calls):
    gs = {json.dumps(norm_call(c), sort_keys=True) for c in gold_calls}
    ps = {json.dumps(norm_call(c), sort_keys=True) for c in pred_calls}
    if not gs and not ps: return True, True
    correct = gs == ps
    args_ok = all(any(g["name"] == p["name"] and g["arguments"] == p["arguments"]
                      for p in pred_calls) for g in gold_calls) if pred_calls else False
    return correct, args_ok

# ── Main ──────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print(f"Loading tokenizer {MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print(f"Loading model {MODEL} ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
).to(device)
model.eval()
print("Model loaded")

results = collections.defaultdict(lambda: {"sel": [], "args": [], "strict": []})
held_results = collections.defaultdict(lambda: {"sel": [], "args": [], "strict": []})

t0 = time.time()
for i in range(0, len(TEST), BATCH):
    batch = TEST[i:i + BATCH]
    prompts = [render_prompt(row) for row in batch]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    for j, row in enumerate(batch):
        input_len = inputs["input_ids"].shape[1]
        gen = tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True)
        pred_calls = parse_tool_calls(gen)
        gold_calls = row.get("gold_calls", [])
        category = row.get("category", "unknown")
        held = row.get("held_out_tool", False)

        correct, args_ok = match_score(gold_calls, pred_calls)
        target = held_results if held else results
        target[category]["sel"].append(correct)
        target[category]["args"].append(args_ok)
        target[category]["strict"].append(correct and args_ok)

        if DUMP and j < DUMP:
            print(f"\n--- Row {i + j} ({category}) ---")
            print(f"GOLD: {gold_calls}")
            print(f"PRED: {pred_calls}")
            print(f"CORRECT: {correct}")

    elapsed = time.time() - t0
    pct = (i + len(batch)) / len(TEST) * 100
    rate = (i + len(batch)) / elapsed if elapsed > 0 else 0
    print(f"  [{i + len(batch)}/{len(TEST)}] {pct:.0f}%  {rate:.2f} rows/s  {elapsed:.0f}s elapsed", end="\r")

print("\n" + "=" * 60)
print("STANDARD (non-held-out) RESULTS")
print("=" * 60)
all_sel, all_args, all_strict = [], [], []
for cat in sorted(results):
    r = results[cat]
    n = len(r["sel"])
    sel = sum(r["sel"]) / n * 100 if n else 0
    args = sum(r["args"]) / n * 100 if n else 0
    strict = sum(r["strict"]) / n * 100 if n else 0
    all_sel.extend(r["sel"]); all_args.extend(r["args"]); all_strict.extend(r["strict"])
    print(f"  {cat:20s}  selection={sel:.1f}  arguments={args:.1f}  strict={strict:.1f}  n={n}")

n = len(all_sel)
print(f"\n  {'AVERAGE':20s}  selection={sum(all_sel)/n*100:.1f}  arguments={sum(all_args)/n*100:.1f}  strict={sum(all_strict)/n*100:.1f}  n={n}")

print("\n" + "=" * 60)
print("HELD-OUT TOOL RESULTS")
print("=" * 60)
hs, ha, hst = [], [], []
for cat in sorted(held_results):
    r = held_results[cat]
    n = len(r["sel"])
    sel = sum(r["sel"]) / n * 100 if n else 0
    args = sum(r["args"]) / n * 100 if n else 0
    strict = sum(r["strict"]) / n * 100 if n else 0
    hs.extend(r["sel"]); ha.extend(r["args"]); hst.extend(r["strict"])
    print(f"  {cat:20s}  selection={sel:.1f}  arguments={args:.1f}  strict={strict:.1f}  n={n}")

if hs:
    n = len(hs)
    print(f"\n  {'HELD AVG':20s}  selection={sum(hs)/n*100:.1f}  arguments={sum(ha)/n*100:.1f}  strict={sum(hst)/n*100:.1f}  n={n}")

print(f"\nTotal time: {time.time() - t0:.0f}s")
