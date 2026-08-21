# /// script
# dependencies = [
#   "huggingface_hub>=0.24",
# ]
# ///

"""P4.4 — Weekly SakThai ops report.

Aggregates the last N days of HF Jobs into a summary suitable for a Slack
post, an email, or a git commit into `Nanthasit/eval_results/ops/`.

Runs on cpu-basic and finishes in seconds. Requires a token with
`inference.jobs.read` scope (a normal read token works).

Schedule with:

    hf jobs scheduled uv \\
        --schedule '0 8 * * MON' \\
        --flavor cpu-basic \\
        --secrets HF_TOKEN \\
        https://huggingface.co/datasets/Nanthasit/hf-cli-jobs-uv-run-scripts/resolve/main/weekly_ops_report.py

Or, to just print to stdout for a one-off:

    HF_TOKEN=... python weekly_ops_report.py --days 7
"""

import argparse
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from huggingface_hub import HfApi


GPU_FLAVORS = {
    "a100-large", "a10g-large", "a10g-small", "l4x4", "l4x1",
    "h100", "h200", "t4-medium", "t4-small", "zero-a10g",
}


def _fmt_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60}m"


def _bucket_script(command: list[str]) -> str:
    """Compress a job command to a short 'script name' bucket."""
    joined = " ".join(command)[:200]
    for marker in ("/data/", "/tmp/", "scripts/"):
        idx = joined.find(marker)
        if idx >= 0:
            tail = joined[idx + len(marker):]
            return tail.split()[0][:60]
    return joined.split()[0][:60] if joined else "(empty)"


def collect(api: HfApi, since: datetime) -> list[dict]:
    all_jobs = list(api.list_jobs(all=True, limit=None))
    fresh = []
    for j in all_jobs:
        created = getattr(j, "created_at", None)
        if created is None:
            continue
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if created >= since:
            fresh.append(j)
    return fresh


def summarize(jobs: list[dict]) -> str:
    n = len(jobs)
    if n == 0:
        return "No jobs in window."

    stage = Counter(getattr(j.status, "stage", "UNKNOWN") for j in jobs)
    flavor = Counter(getattr(j, "flavor", "?") for j in jobs)
    gpu_seconds = defaultdict(int)
    script_stats = defaultdict(lambda: {"total": 0, "error": 0})
    wasted_gpu_sec = 0

    for j in jobs:
        f = getattr(j, "flavor", "?")
        dur = 0
        d = getattr(j, "durations", None) or {}
        if isinstance(d, dict):
            dur = int(d.get("runningSecs") or 0)
        elif hasattr(d, "runningSecs"):
            dur = int(getattr(d, "runningSecs") or 0)
        if f in GPU_FLAVORS:
            gpu_seconds[f] += dur

        script = _bucket_script(getattr(j, "command", []) or [])
        script_stats[script]["total"] += 1
        st = getattr(j.status, "stage", "UNKNOWN")
        if st in ("ERROR", "CANCELED"):
            script_stats[script]["error"] += 1
            if f in GPU_FLAVORS:
                wasted_gpu_sec += dur

    error_rate = 100.0 * (stage.get("ERROR", 0) + stage.get("CANCELED", 0)) / n

    lines = []
    lines.append(f"# SakThai weekly ops report")
    lines.append(f"Window: {n} jobs")
    lines.append("")
    lines.append(f"**Error+Canceled rate:** {error_rate:.1f}%   (target: <15%)")
    lines.append(f"**Wasted GPU time (ERROR/CANCELED on GPU flavors):** {_fmt_duration(wasted_gpu_sec)}")
    lines.append("")

    lines.append("## Status breakdown")
    for s, c in stage.most_common():
        lines.append(f"- {s}: {c}")
    lines.append("")

    lines.append("## GPU time by flavor (running seconds)")
    for f, secs in sorted(gpu_seconds.items(), key=lambda x: -x[1]):
        lines.append(f"- {f}: {_fmt_duration(secs)}")
    if not gpu_seconds:
        lines.append("- (no GPU jobs in window)")
    lines.append("")

    lines.append("## Top 10 failing scripts")
    ranked = sorted(
        ((s, v["error"], v["total"]) for s, v in script_stats.items() if v["error"] > 0),
        key=lambda x: (-x[1], -x[2]),
    )
    if not ranked:
        lines.append("- (no failures)")
    else:
        for script, errs, tot in ranked[:10]:
            lines.append(f"- `{script}`: {errs}/{tot} failed ({100.0 * errs / tot:.0f}%)")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=7, help="Window in days (default: 7)")
    args = p.parse_args()

    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    jobs = collect(api, since)
    print(summarize(jobs))


if __name__ == "__main__":
    main()
