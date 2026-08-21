# /// script
# dependencies = [
#   "datasets>=2.19",
#   "huggingface_hub>=0.24",
# ]
# ///

"""P6.3 — Compute category-balance metadata for a SakThai combined-* dataset.

Classifies every row into one of:
    - simple       — exactly one assistant turn that emits one tool call
    - parallel     — an assistant turn that emits >1 tool call
    - irrelevance  — no assistant tool call in the entire conversation
    - multi_turn   — 2+ assistant turns with tool calls
    - other        — anything the heuristic can't classify

Prints a YAML block ready to paste into the dataset's frontmatter under
`sakthai_composition:`.

    python dataset_composition_stats.py Nanthasit/sakthai-combined-v11
    python dataset_composition_stats.py Nanthasit/sakthai-combined-v12 \\
        --data-files data/train.jsonl        # for datasets whose parquet is broken
"""

import argparse
import json
import re
import sys
from collections import Counter

from datasets import load_dataset


_TC_RE = re.compile(r"<tool_call>", re.IGNORECASE)


def _num_tool_calls(text: str) -> int:
    if not text:
        return 0
    return len(_TC_RE.findall(text))


def _content_str(c) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    try:
        return json.dumps(c, ensure_ascii=False)
    except Exception:
        return str(c)


def classify_row(row: dict) -> str:
    messages = row.get("messages")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            return "other"
    if not isinstance(messages, list):
        return "other"

    assistant_turns_with_calls = 0
    parallel_seen = False
    for m in messages:
        if m.get("role") != "assistant":
            continue
        n = _num_tool_calls(_content_str(m.get("content")))
        # tool_calls may also live under a structured field
        tc = m.get("tool_calls")
        if isinstance(tc, list):
            n = max(n, len(tc))
        if n >= 2:
            parallel_seen = True
        if n >= 1:
            assistant_turns_with_calls += 1

    if assistant_turns_with_calls == 0:
        return "irrelevance"
    if parallel_seen:
        return "parallel"
    if assistant_turns_with_calls >= 2:
        return "multi_turn"
    return "simple"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", help="HF dataset repo id, e.g. Nanthasit/sakthai-combined-v11")
    p.add_argument("--split", default="train")
    p.add_argument("--data-files", default=None,
                   help="Override for datasets whose parquet auto-conversion is broken (e.g. 'data/train.jsonl').")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Limit for smoke-testing on huge datasets.")
    args = p.parse_args()

    kwargs = {"split": args.split}
    if args.data_files:
        kwargs["data_files"] = args.data_files
    ds = load_dataset(args.dataset, **kwargs)
    if args.max_rows:
        ds = ds.select(range(min(args.max_rows, len(ds))))

    counts = Counter()
    for row in ds:
        counts[classify_row(row)] += 1

    total = sum(counts.values())
    if total == 0:
        raise SystemExit("[stats] no rows loaded")

    pct = {k: 100.0 * counts[k] / total for k in counts}

    print(f"# Composition for {args.dataset} ({total} rows, split={args.split})", file=sys.stderr)
    for k in sorted(counts):
        print(f"#   {k:12s}  {counts[k]:6d}  {pct[k]:5.1f}%", file=sys.stderr)
    print("", file=sys.stderr)

    print("# Paste into the dataset's README frontmatter:")
    print("sakthai_composition:")
    for k in ("simple", "parallel", "irrelevance", "multi_turn", "other"):
        if counts.get(k, 0):
            print(f"  {k}: {pct[k]:.1f}")
    print(f"  total_rows: {total}")


if __name__ == "__main__":
    main()
