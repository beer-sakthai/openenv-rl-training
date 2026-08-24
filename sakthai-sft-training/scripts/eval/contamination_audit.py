# /// script
# dependencies = [
#   "datasets>=2.19",
#   "huggingface_hub>=0.24",
# ]
# ///

"""P5.4 — Contamination audit for sakthai-bench-v2.

Checks whether ANY prompt in the benchmark also appears in the training
mix (combined-v6/v7/v10/v11/v12 and any others you pass). If it does, the
`held_out` score is not really held out; the true generalization number is
the score on the *clean* subset.

Uses SHA-256 over a normalized prompt string. Normalization:
    - strip whitespace
    - lowercase
    - if messages field contains a JSON list, canonicalize with sorted keys

Reports per-training-dataset overlap counts and writes a markdown report
to eval_results/contamination-report.md (stdout by default).

    python contamination_audit.py \\
        --bench Nanthasit/sakthai-bench-v2 \\
        --training Nanthasit/sakthai-combined-v6 \\
                   Nanthasit/sakthai-combined-v7 \\
                   Nanthasit/sakthai-combined-v10 \\
                   Nanthasit/sakthai-combined-v11 \\
        --training-with-data-files Nanthasit/sakthai-combined-v12=data/train.jsonl \\
        --out contamination-report.md
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from typing import Iterable

from datasets import load_dataset


def _content_str(c) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    try:
        return json.dumps(c, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(c)


def _first_user_prompt(row: dict) -> str:
    messages = row.get("messages")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            return ""
    if not isinstance(messages, list):
        # some benches store a bare "prompt" field
        return _content_str(row.get("prompt") or row.get("query"))
    for m in messages:
        if m.get("role") == "user":
            return _content_str(m.get("content"))
    return ""


def _normalize(s: str) -> str:
    return " ".join(s.strip().lower().split())


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _iter_rows(repo: str, split: str, data_files: str | None) -> Iterable[dict]:
    kwargs = {"split": split}
    if data_files:
        kwargs["data_files"] = data_files
    return load_dataset(repo, **kwargs)


def build_hash_index(repo: str, split: str, data_files: str | None) -> dict[str, int]:
    idx: dict[str, int] = {}
    for i, row in enumerate(_iter_rows(repo, split, data_files)):
        h = _hash(_normalize(_first_user_prompt(row)))
        idx.setdefault(h, i)
    return idx


def build_bench_index(repo: str, split: str) -> dict[str, dict]:
    idx: dict[str, dict] = {}
    for i, row in enumerate(_iter_rows(repo, split, None)):
        prompt = _first_user_prompt(row)
        h = _hash(_normalize(prompt))
        cat = row.get("category") or row.get("subset") or "unknown"
        idx.setdefault(h, {"row_index": i, "category": cat, "prompt_preview": prompt[:120]})
    return idx


def get_training_specs(args: argparse.Namespace) -> list[tuple[str, str | None]]:
    training_specs = [(r, None) for r in args.training]
    for spec in args.training_with_data_files:
        if "=" not in spec:
            raise SystemExit(f"[audit] bad spec {spec!r}, expected REPO=data/path.jsonl")
        repo, df = spec.split("=", 1)
        training_specs.append((repo, df))
    return training_specs


def compute_overlaps(training_specs: list[tuple[str, str | None]], bench_idx: dict[str, dict], training_split: str) -> tuple[dict[str, list[str]], dict[str, int], set[str]]:
    per_dataset_overlap: dict[str, list[str]] = {}
    all_overlap_by_cat = defaultdict(int)
    total_overlap_hashes = set()

    for repo, df in training_specs:
        print(f"[audit] indexing training {repo} (data_files={df})...", file=sys.stderr)
        try:
            train_idx = build_hash_index(repo, training_split, df)
        except Exception as e:
            print(f"[audit] FAILED to load {repo}: {e}", file=sys.stderr)
            continue
        overlap = sorted(set(train_idx) & set(bench_idx))
        per_dataset_overlap[repo] = overlap
        for h in overlap:
            total_overlap_hashes.add(h)
        for h in overlap:
            all_overlap_by_cat[bench_idx[h]["category"]] += 1
        print(f"[audit]   {repo}: {len(overlap)} overlaps", file=sys.stderr)

    return per_dataset_overlap, all_overlap_by_cat, total_overlap_hashes


def generate_report(bench: str, bench_idx: dict[str, dict], per_dataset_overlap: dict[str, list[str]], all_overlap_by_cat: dict[str, int], total_overlap_hashes: set[str], out: str) -> None:
    lines = []
    lines.append(f"# Contamination audit — {bench}")
    lines.append("")
    lines.append(f"- Bench prompts (unique hashes): **{len(bench_idx)}**")
    lines.append(f"- Distinct bench prompts also present in the training mix: "
                 f"**{len(total_overlap_hashes)}** "
                 f"({100.0 * len(total_overlap_hashes) / max(1, len(bench_idx)):.1f}%)")
    lines.append("")

    lines.append("## Per training-dataset overlap")
    lines.append("| Training dataset | Overlapping bench rows |")
    lines.append("|---|---:|")
    for repo, hashes in per_dataset_overlap.items():
        lines.append(f"| `{repo}` | {len(hashes)} |")
    lines.append("")

    if all_overlap_by_cat:
        lines.append("## Overlap by bench category")
        lines.append("| Category | Contaminated rows |")
        lines.append("|---|---:|")
        for cat, n in sorted(all_overlap_by_cat.items()):
            lines.append(f"| {cat} | {n} |")
        lines.append("")

    lines.append("## Contaminated bench-row previews (up to 20)")
    shown = 0
    for h in sorted(total_overlap_hashes):
        if shown >= 20:
            break
        b = bench_idx[h]
        lines.append(f"- **row {b['row_index']}** (`{b['category']}`): "
                     f"`{b['prompt_preview']}`")
        shown += 1

    lines.append("")
    lines.append("## Conclusion")
    if len(total_overlap_hashes) == 0:
        lines.append("**No overlap detected.** The `held_out` bench score can be "
                     "trusted as an out-of-sample generalization number.")
    else:
        n = len(total_overlap_hashes)
        pct = 100.0 * n / len(bench_idx)
        lines.append(f"**Overlap detected on {n} rows ({pct:.1f}%).** The "
                     f"reported `held_out` score is inflated by whatever the "
                     f"model memorized from the training set. The true "
                     f"generalization number is the score computed on the "
                     f"clean subset (bench minus these row indices).")

    out_text = "\n".join(lines)
    if out == "-":
        print(out_text)
    else:
        from pathlib import Path
        Path(out).write_text(out_text)
        print(f"[audit] wrote {out}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True, help="Benchmark dataset repo id")
    p.add_argument("--bench-split", default="train",
                   help="Split within the bench dataset (default: train)")
    p.add_argument("--training", nargs="*", default=[],
                   help="Training dataset repo ids (default parquet)")
    p.add_argument("--training-with-data-files", nargs="*", default=[],
                   help="Entries of the form REPO=data/path.jsonl for datasets whose parquet is broken")
    p.add_argument("--training-split", default="train")
    p.add_argument("--out", default="-", help="Output markdown path or '-' for stdout")
    args = p.parse_args()

    print(f"[audit] indexing bench {args.bench}...", file=sys.stderr)
    bench_idx = build_bench_index(args.bench, args.bench_split)
    print(f"[audit] bench has {len(bench_idx)} unique prompt hashes", file=sys.stderr)

    training_specs = get_training_specs(args)
    per_dataset_overlap, all_overlap_by_cat, total_overlap_hashes = compute_overlaps(
        training_specs, bench_idx, args.training_split
    )
    generate_report(
        args.bench, bench_idx, per_dataset_overlap, all_overlap_by_cat, total_overlap_hashes, args.out
    )


if __name__ == "__main__":
    main()
