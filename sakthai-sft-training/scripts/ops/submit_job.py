# /// script
# dependencies = [
#   "huggingface_hub>=0.24",
# ]
# ///

"""P4.2 + P4.3 — Guarded HF Jobs submit wrapper.

Enforces two policies at submit time (before a single dollar is spent):

  P4.3 (tagging): every job must declare --tag from the allowed set, so
    dashboards can separate real training from experiments and one-offs.

  P4.2 (debug on GPU): commands matching known debug patterns
    (echo hello, python -c print, test_job.py, hello-world equivalents)
    are FORCED down to cpu-basic regardless of the requested flavor.
    You cannot bill an a100 to prove Python still prints.

Use as a wrapper instead of `hf jobs uv run` directly:

    python submit_job.py \\
        --tag prod-train \\
        --flavor a100-large \\
        --script https://huggingface.co/datasets/Nanthasit/hf-cli-jobs-uv-run-scripts/resolve/main/train.py \\
        --env TRAIN_ENV=browsergym --env TRAIN_BASE=Nanthasit/sakthai-context-7b-tools

Prints the fully-resolved `hf jobs uv run` command it would execute; add
--dry-run to stop before submission. Without --dry-run it calls the HF API
via huggingface_hub.
"""

import argparse
import os
import re
import sys
from typing import Optional

from huggingface_hub import HfApi

ALLOWED_TAGS = {"prod-train", "prod-eval", "debug", "experiment"}
GPU_FLAVORS = {
    "a100-large", "a10g-large", "a10g-small", "l4x4", "l4x1",
    "h100", "h200", "t4-medium", "t4-small", "zero-a10g",
}

# Anything matching one of these is a debug/hello-world job.
DEBUG_PATTERNS = [
    re.compile(r"\becho\s+[\"']?hello", re.IGNORECASE),
    re.compile(r"python\s+-c\s+[\"']?print\(", re.IGNORECASE),
    re.compile(r"\btest_job\.py\b"),
    re.compile(r"\bhello[_-]?world\b", re.IGNORECASE),
    re.compile(r"\bsmoke[_-]?test\.py\b"),   # smoke-tests should be cpu-basic too
    re.compile(r"\bduckdb\s+-c\s+select\s+hello", re.IGNORECASE),
]


def looks_like_debug(script_or_cmd: str) -> Optional[str]:
    for pat in DEBUG_PATTERNS:
        if pat.search(script_or_cmd):
            return pat.pattern
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, choices=sorted(ALLOWED_TAGS),
                   help="Job purpose tag. Enforced.")
    p.add_argument("--flavor", required=True,
                   help="Requested hardware. May be down-graded to cpu-basic "
                        "if the command matches a debug pattern.")
    p.add_argument("--script", required=True,
                   help="URL of the uv script to run (raw file URL on the Hub).")
    p.add_argument("--env", action="append", default=[],
                   help="KEY=VALUE, repeatable.")
    p.add_argument("--secrets", action="append", default=[],
                   help="Secret name to pass through, repeatable.")
    p.add_argument("--timeout", default="1h",
                   help="Job wall-clock timeout (e.g. '30m', '2h').")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the resolved plan without submitting.")
    args = p.parse_args()

    # P4.2 — debug pattern check
    matched = looks_like_debug(args.script)
    resolved_flavor = args.flavor
    if matched and args.flavor in GPU_FLAVORS:
        print(f"[submit] DEBUG PATTERN matched ({matched!r})", file=sys.stderr)
        print(f"[submit] FORCING flavor: {args.flavor} -> cpu-basic", file=sys.stderr)
        resolved_flavor = "cpu-basic"

    env_map = {}
    for kv in args.env:
        if "=" not in kv:
            raise SystemExit(f"[submit] bad --env {kv!r}, expected KEY=VALUE")
        k, v = kv.split("=", 1)
        env_map[k] = v

    plan = {
        "tag": args.tag,
        "flavor": resolved_flavor,
        "script": args.script,
        "env": env_map,
        "secrets": args.secrets,
        "timeout": args.timeout,
    }
    print("[submit] resolved plan:")
    for k, v in plan.items():
        print(f"    {k}: {v}")

    if args.dry_run:
        print("[submit] --dry-run set; not submitting")
        return

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    # NOTE: exact call shape depends on huggingface_hub version; keep this
    # as a thin adapter around whatever run_uv_job / jobs API is current.
    job = api.run_uv_job(
        script=args.script,
        flavor=resolved_flavor,
        env=env_map,
        secrets=args.secrets,
        timeout=args.timeout,
        # Labels/tags syntax: whatever the current API accepts. If your
        # huggingface_hub is older, drop this kwarg and set the label
        # via env var like SAKTHAI_TAG=prod-train that weekly_ops_report.py
        # can bucket on.
        labels={"sakthai_tag": args.tag},
    )
    print(f"[submit] submitted: id={getattr(job, 'id', '?')}")


if __name__ == "__main__":
    main()
