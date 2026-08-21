# /// script
# dependencies = ["transformers>=4.44", "datasets", "accelerate", "torch", "huggingface_hub", "peft"]
# ///
"""sakthai-bench-v2 evaluator, patched with bare-PEFT-adapter support.

This is Nanthasit/sakthai-bench-v2's eval_bench.py, unmodified except for
load_model()/_get_tokenizer() (see PATCH markers below). Every rendering and
scoring function is byte-identical to the canonical script so results stay
comparable to prior -merged runs. Not pushed to the canonical repo -- run as
a standalone job until the adapter-loading path is reviewed.

Usage (HF Jobs) -- same env vars as the canonical script:
    hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \
      --env SAK_MODELS=Nanthasit/sakthai-context-0.5b-tools \
      --env SAK_UPLOAD_TO=Nanthasit/sakthai-bench-v2 \
      ./eval_bench_peft.py

Env vars: identical to eval_bench.py (SAK_MODELS, SAK_BENCH, SAK_TOKENIZER,
SAK_UPLOAD_TO, SAK_DUMP, SAK_DUMP_ROWS, SAK_BATCH, SAK_FORCE_CPU, SAK_DTYPE).
"""

import collections
import gc
import json
import os
import re
import time
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

BENCH = os.environ.get("SAK_BENCH", "Nanthasit/sakthai-bench-v2")
MODELS = [m.strip() for m in os.environ.get("SAK_MODELS", "").split(",") if m.strip()]
TOKENIZER_ID = os.environ.get("SAK_TOKENIZER", "")  # empty = per-model fallback
UPLOAD_TO = os.environ.get("SAK_UPLOAD_TO", "")
DUMP = int(os.environ.get("SAK_DUMP", "0"))
DUMP_ROWS = int(os.environ.get("SAK_DUMP_ROWS", "0"))
BATCH = int(os.environ.get("SAK_BATCH", "16"))
FORCE_CPU = int(os.environ.get("SAK_FORCE_CPU", "0"))
DTYPE = os.environ.get("SAK_DTYPE", "").strip()  # "" = auto per device
assert MODELS, "set SAK_MODELS to a comma-separated list of model ids"

_GLOBAL_TOK = None
if TOKENIZER_ID:
    _GLOBAL_TOK = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    if _GLOBAL_TOK.pad_token is None:
        _GLOBAL_TOK.pad_token = _GLOBAL_TOK.eos_token
    _GLOBAL_TOK.padding_side = "left"


# ── Renderer: verbatim from eval_bench.py, unchanged ──────────────────────
def _text(c):
    return (
        ""
        if c is None
        else (c if isinstance(c, str) else json.dumps(c, ensure_ascii=False))
    )


def _tools_block(tools):
    if not tools:
        return ""
    sigs = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
    return (
        "\n\n# Tools\n\nYou may call one or more functions. Signatures are within "
        "<tools></tools>:\n<tools>\n" + sigs + "\n</tools>\n\nFor each call return:\n"
        '<tool_call>\n{"name": <name>, "arguments": <json>}\n</tool_call>'
    )


def _assistant_body(m):
    parts = []
    content = _text(m.get("content"))
    if content:
        parts.append(content)
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", tc)
        a = fn.get("arguments", "{}")
        if not isinstance(a, str):
            a = json.dumps(a, ensure_ascii=False)
        parts.append(
            f'<tool_call>\n{{"name": "{fn.get("name", "")}", "arguments": {a}}}\n</tool_call>'
        )
    return "\n".join(parts)


def _render_msg(m, tools_sys):
    r = m.get("role")
    if r == "system":
        return (
            "<|im_start|>system\n"
            + _text(m.get("content"))
            + _tools_block(tools_sys)
            + "<|im_end|>\n"
        )
    if r == "user":
        return "<|im_start|>user\n" + _text(m.get("content")) + "<|im_end|>\n"
    if r == "tool":
        return (
            "<|im_start|>user\n<tool_response>\n"
            + _text(m.get("content"))
            + "\n</tool_response><|im_end|>\n"
        )
    if r == "assistant":
        return "<|im_start|>assistant\n" + _assistant_body(m) + "<|im_end|>\n"
    return ""


def render_prompt(messages, tools):
    out = []
    if not (messages and messages[0].get("role") == "system") and tools:
        out.append(
            "<|im_start|>system\nYou are a helpful assistant."
            + _tools_block(tools)
            + "<|im_end|>\n"
        )
    for i, m in enumerate(messages):
        out.append(
            _render_msg(m, tools if (i == 0 and m.get("role") == "system") else None)
        )
    out.append("<|im_start|>assistant\n")
    return "".join(out)


_TC = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_calls(text):
    out = []
    for m in _TC.findall(text):
        try:
            d = json.loads(m)
        except Exception:
            continue
        if d.get("name"):
            a = d.get("arguments")
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    a = {"__unparsed__": a}
            out.append(
                {"name": d["name"], "arguments": a if isinstance(a, dict) else {}}
            )
    return out


def is_degenerate(raw):
    s = "".join(raw.split())
    if len(s) < 20:
        return False
    return max(collections.Counter(s).values()) / len(s) >= 0.9


# ── Scoring: verbatim from eval_bench.py, unchanged ────────────────────────
def norm(v):
    if isinstance(v, str):
        s = v.strip()
        try:
            return norm(json.loads(s))
        except Exception:
            return s.lower()
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        return {k: norm(x) for k, x in sorted(v.items())}
    if isinstance(v, list):
        return [norm(x) for x in v]
    return v


def args_match(gold, pred):
    return norm(gold) == norm(pred)


def selection_ok(cat, gold_names, pred_names):
    if cat.startswith("irrelevance"):
        return len(pred_names) == 0
    if cat == "simple":
        return bool(gold_names) and gold_names[0] in pred_names
    return not (Counter(gold_names) - Counter(pred_names))


def _match_all(gold_calls, pred):
    remaining = list(pred)
    for g in gold_calls:
        for i, c in enumerate(remaining):
            if c["name"] == g["name"] and args_match(g["arguments"], c["arguments"]):
                del remaining[i]
                break
        else:
            return False
    return True


def arguments_ok(cat, gold_calls, pred):
    if cat.startswith("irrelevance"):
        return None
    if cat == "simple":
        if not gold_calls:
            return False
        g = gold_calls[0]
        return any(
            c["name"] == g["name"] and args_match(g["arguments"], c["arguments"])
            for c in pred
        )
    return _match_all(gold_calls, pred)


def args_partial(gold_calls, pred):
    if not gold_calls:
        return None
    tot = 0.0
    for g in gold_calls:
        cand = [c for c in pred if c["name"] == g["name"]]
        if not cand:
            continue
        best = 0.0
        for c in cand:
            ga, pa = g["arguments"], c["arguments"]
            if not ga:
                best = max(best, 1.0)
                continue
            hit = sum(1 for k, v in ga.items() if k in pa and norm(v) == norm(pa[k]))
            best = max(best, hit / len(ga))
        tot += best
    return tot / len(gold_calls)


def classify_arg_errors(gold_calls, pred):
    errors = []
    for g in gold_calls:
        cand = [c for c in pred if c["name"] == g["name"]]
        if not cand:
            errors.append(
                {
                    "name": g["name"],
                    "type": "missing_call",
                    "detail": "no matching call found",
                }
            )
            continue
        c = cand[0]
        ga, pa = g["arguments"], c["arguments"]
        for k, v in ga.items():
            if k not in pa:
                errors.append(
                    {
                        "name": g["name"],
                        "key": k,
                        "type": "missing_key",
                        "gold": v,
                        "pred": None,
                    }
                )
            elif norm(v) != norm(pa[k]):
                etype = "type_mismatch" if not isinstance(v, type(pa[k])) else "wrong_value"
                errors.append(
                    {
                        "name": g["name"],
                        "key": k,
                        "type": etype,
                        "gold": v,
                        "pred": pa[k],
                    }
                )
        for k in pa:
            if k not in ga:
                errors.append(
                    {
                        "name": g["name"],
                        "key": k,
                        "type": "extra_key",
                        "gold": None,
                        "pred": pa[k],
                    }
                )
    return errors


# ── Load bench rows once: verbatim from eval_bench.py, unchanged ──────────
TEST = load_dataset(BENCH, split="test")
CATS = ("simple", "parallel", "irrelevance_tools", "irrelevance_no_tools")

ROWS = []
for ex in TEST:
    _i = ex.get("turn_index")
    if _i is None:
        _i = next(
            (i for i, mm in enumerate(ex["messages"]) if mm.get("role") == "assistant"),
            None,
        )
    if _i is None:
        continue
    ROWS.append(
        {
            "prompt": render_prompt(ex["messages"][:_i], ex.get("tools") or None),
            "cat": ex["category"],
            "gold_calls": ex.get("gold_calls") or [],
            "held_out": bool(ex.get("held_out_tool")),
            "multi_turn": bool(ex.get("multi_turn")),
            "in_v1": bool(ex.get("in_v1", True)),
        }
    )
print(f"{len(ROWS)} bench rows, batch size {BATCH}")


def generate_all(m, tok):
    outs = [None] * len(ROWS)
    order = sorted(range(len(ROWS)), key=lambda i: len(ROWS[i]["prompt"]), reverse=True)
    for s in range(0, len(order), BATCH):
        chunk = order[s: s + BATCH]
        enc = tok(
            [ROWS[i]["prompt"] for i in chunk],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(m.device)
        with torch.no_grad():
            out = m.generate(
                **enc,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        for j, i in enumerate(chunk):
            outs[i] = tok.decode(gen[j], skip_special_tokens=True)
    return outs


# ── Precision selection: verbatim from eval_bench.py, unchanged ───────────
_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
    "full": "float32",
}


def dtype_candidates(device, bf16_supported, forced=None):
    if forced:
        key = str(forced).strip().lower()
        if key not in _ALIASES:
            raise ValueError(
                f"unknown SAK_DTYPE {forced!r}; expected one of {sorted(set(_ALIASES))}"
            )
        return [_ALIASES[key]]
    if device != "cuda":
        return ["float32"]
    if bf16_supported:
        return ["bfloat16", "float16", "float32"]
    return ["float16", "float32"]


# ── PATCH START: bare-PEFT-adapter support ─────────────────────────────────
def _peft_base_model(repo_id):
    """Return the adapter's base_model_name_or_path if repo_id is a bare PEFT
    adapter repo (has adapter_config.json, no full model weights), else None.
    """
    try:
        from huggingface_hub import hf_hub_download

        cfg_path = hf_hub_download(repo_id, "adapter_config.json")
    except Exception:
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise ValueError(
            f"{repo_id}: adapter_config.json has no base_model_name_or_path"
        )
    return base


# ── PATCH END ────────────────────────────────────────────────────────────


def load_model(repo_id):
    """Load a model in the best precision this device can actually run.

    PATCH: if repo_id is a bare PEFT adapter, load its base model first and
    attach the adapter with PeftModel.from_pretrained -- the original script
    only ever called AutoModelForCausalLM.from_pretrained(repo_id, ...)
    directly, which errors on adapter-only repos (no full model weights),
    so sakthai-context-*-tools was never actually loadable by eval_bench.py.
    """
    device = "cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu"
    bf16 = bool(device == "cuda" and torch.cuda.is_bf16_supported())
    names = dtype_candidates(device, bf16, DTYPE or None)
    base_id = _peft_base_model(repo_id)  # PATCH: None for merged/full-weight repos

    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(
            f"  device: {torch.cuda.get_device_name(0)} "
            f"({total / 1e9:.1f} GB total, {free / 1e9:.1f} GB free, bf16={bf16})"
        )
    last = None
    for name in names:
        try:
            load_id = base_id or repo_id
            print(
                f"  loading {load_id} in {name} on {device}"
                + (f" (+ adapter {repo_id})" if base_id else "")
                + "..."
            )
            m = AutoModelForCausalLM.from_pretrained(
                load_id,
                torch_dtype=getattr(torch, name),
                device_map=device if device == "cuda" else None,
            )
            if base_id:  # PATCH: attach adapter, keep as PeftModel (no merge)
                from peft import PeftModel

                m = PeftModel.from_pretrained(m, repo_id)
            return m.to("cpu") if device == "cpu" else m
        except Exception as e:
            last = e
            print(f"    {name} on {device} failed: {type(e).__name__}: {e}")
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    raise RuntimeError(f"could not load {repo_id} in any of {names}") from last


def _process_eval_rows(raws):
    sel = collections.defaultdict(lambda: [0, 0])
    arg = collections.defaultdict(lambda: [0, 0])
    strict = collections.defaultdict(lambda: [0, 0])
    ho = [0, 0]
    dead = [0]
    dumped = 0
    partial = []
    slices = {"multi_turn": [0, 0], "single_turn": [0, 0], "in_v1": [0, 0]}
    all_errors = []

    for row, raw in zip(ROWS, raws):
        cat, gold_calls = row["cat"], row["gold_calls"]
        pred = parse_calls(raw)
        pred_names = [c["name"] for c in pred]
        gold_names = [c["name"] for c in gold_calls]

        if DUMP and dumped < DUMP and gold_names:
            dumped += 1
            print(f"\n--- raw generation [{cat}] gold={gold_names} parsed={pred_names}")
            print(repr(raw[:400]))

        if is_degenerate(raw):
            dead[0] += 1
            ok_sel, ok_args = False, (None if cat.startswith("irrelevance") else False)
        else:
            ok_sel = bool(selection_ok(cat, gold_names, pred_names))
            ok_args = (
                None
                if cat.startswith("irrelevance")
                else bool(ok_sel and arguments_ok(cat, gold_calls, pred))
            )

        if not ok_sel and gold_calls and DUMP_ROWS:
            errors = classify_arg_errors(gold_calls, pred)
            all_errors.extend(errors)

        if DUMP_ROWS and gold_calls:
            status = "PASS" if ok_sel else "FAIL"
            print(f"  [{cat}] {status}: gold={gold_names} pred={pred_names}")

        sel[cat][0] += int(ok_sel)
        sel[cat][1] += 1
        if ok_args is not None:
            arg[cat][0] += int(ok_args)
            arg[cat][1] += 1
            strict[cat][0] += int(ok_sel and ok_args)
            strict[cat][1] += 1
        if row["held_out"]:
            ho[0] += int(ok_sel)
            ho[1] += 1
        k = "multi_turn" if row["multi_turn"] else "single_turn"
        slices[k][0] += int(ok_sel)
        slices[k][1] += 1
        if row["in_v1"]:
            slices["in_v1"][0] += int(ok_sel)
            slices["in_v1"][1] += 1
        if gold_calls and not is_degenerate(raw):
            pp = args_partial(gold_calls, pred)
            if pp is not None:
                partial.append(pp)

    del m
    gc.collect()
    torch.cuda.empty_cache()
    return sel, arg, strict, ho, dead, partial, slices, all_errors


def _build_and_print_summary(repo_id, t0, runtime, stats):
    sel, arg, strict, ho, dead, partial, slices, all_errors = stats

    def agg(d):
        p = sum(v[0] for v in d.values())
        t = sum(v[1] for v in d.values())
        return p, t, (100 * p / t if t else None)

    err_summary = {}
    for e in all_errors:
        typ = e["type"]
        err_summary[typ] = err_summary.get(typ, 0) + 1

    res = {
        "model": repo_id,
        "seconds": round(time.time() - t0, 1),
        "runtime": runtime,
        "selection": {c: {"pass": sel[c][0], "total": sel[c][1]} for c in CATS},
        "arguments": {
            c: {"pass": arg[c][0], "total": arg[c][1]} for c in CATS if arg[c][1]
        },
        "held_out_tools": {"pass": ho[0], "total": ho[1]},
        "degenerate_outputs": dead[0],
        "overall_selection": agg(sel)[2],
        "overall_arguments": agg(arg)[2],
        "overall_strict": agg(strict)[2],
        "arguments_partial": (100 * sum(partial) / len(partial)) if partial else None,
        "slices": {k: {"pass": v[0], "total": v[1]} for k, v in slices.items() if v[1]},
    }
    if err_summary:
        res["error_types"] = err_summary

    print(
        f"\n=== {repo_id}  ({res['seconds']}s, {runtime['dtype']} on {runtime['device']}) ==="
    )
    print(f"{'category':<22}{'selection':>12}{'arguments':>12}")
    for c in CATS:
        s_, st = sel[c]
        a_, at = arg[c]
        sa = f"{100*s_/st:5.1f}%" if st else "   n/a"
        aa = f"{100*a_/at:5.1f}%" if at else "     —"
        print(f"{c:<22}{sa:>12}{aa:>12}")
    print(f"{'OVERALL':<22}{agg(sel)[2]:11.1f}%{(agg(arg)[2] or 0):11.1f}%")
    print(f"{'strict (name+args)':<22}{'':>12}{(agg(strict)[2] or 0):11.1f}%")
    print(f"{'held-out tools':<22}{(100*ho[0]/ho[1] if ho[1] else 0):11.1f}%")
    if partial:
        print(
            f"{'args partial credit':<22}{'':>12}{100*sum(partial)/len(partial):11.1f}%"
        )
    for k in ("single_turn", "multi_turn", "in_v1"):
        pv, tv = slices[k]
        if tv:
            print(f"{'  sel: '+k:<22}{100*pv/tv:11.1f}%   (n={tv})")
    if err_summary:
        print(f"  error types: {err_summary}")
    if dead[0]:
        print(
            f"  !! DEGENERATE OUTPUT on {dead[0]}/{len(ROWS)} rows — this model is broken, "
            f"scores above are not meaningful"
        )
    return res


def evaluate(repo_id):
    t0 = time.time()
    tok = _GLOBAL_TOK or _get_tokenizer(repo_id)
    m = load_model(repo_id)
    m.eval()
    _p = next(m.parameters())
    runtime = {"device": str(_p.device), "dtype": str(_p.dtype).replace("torch.", "")}
    raws = generate_all(m, tok)

    stats = _process_eval_rows(raws)

    del m; gc.collect(); torch.cuda.empty_cache()

    return _build_and_print_summary(repo_id, t0, runtime, stats)


def _get_tokenizer(repo_id):
    """Load a model-specific tokenizer as fallback.

    PATCH: bare adapter repos sometimes ship no tokenizer files -- fall back
    to the adapter's base model tokenizer in that case.
    """
    try:
        tok = AutoTokenizer.from_pretrained(repo_id)
    except Exception as e:
        base = _peft_base_model(repo_id)
        if not base:
            raise
        print(
            f"  {repo_id} has no tokenizer files ({type(e).__name__}); "
            f"falling back to base tokenizer {base}"
        )
        tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


results = []
for repo in MODELS:
    try:
        results.append(evaluate(repo))
    except Exception as e:
        print(f"[FAILED {repo}] {type(e).__name__}: {e}")
        results.append({"model": repo, "error": f"{type(e).__name__}: {e}"})

ok = [r for r in results if "error" not in r]
if ok:
    print("\n" + "=" * 78)
    print(f"{'model':<46}{'sel':>8}{'args':>8}{'strict':>8}{'held':>8}")
    for r in ok:
        h = r["held_out_tools"]
        print(
            f"{r['model'][-45:]:<46}"
            f"{r['overall_selection'] or 0:7.1f}%"
            f"{r['overall_arguments'] or 0:7.1f}%"
            f"{r['overall_strict'] or 0:7.1f}%"
            f"{(100*h['pass']/h['total'] if h['total'] else 0):7.1f}%"
        )

payload = {
    "benchmark": BENCH,
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "scorer": "multiset-selection-v2",
    "results": results,
}
with open("bench_results.json", "w") as f:
    json.dump(payload, f, indent=2)
print("\nwrote bench_results.json")

print("BENCH_RESULTS_JSON_BEGIN")
print(json.dumps(payload, separators=(",", ":")))
print("BENCH_RESULTS_JSON_END")

if UPLOAD_TO:
    try:
        from huggingface_hub import HfApi

        name = f"results/{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
        HfApi().upload_file(
            path_or_fileobj="bench_results.json",
            path_in_repo=name,
            repo_id=UPLOAD_TO,
            repo_type="dataset",
            commit_message=f"bench results: {', '.join(m.split('/')[-1] for m in MODELS)}",
        )
        print(f"uploaded -> {UPLOAD_TO}/{name}")
    except Exception as e:
        print(f"[UPLOAD FAILED] {type(e).__name__}: {e}")
        print(
            "results are NOT lost — recover with tools/results_from_logs.py "
            "(re-run with `--secrets HF_TOKEN` to upload from inside the job)"
        )
