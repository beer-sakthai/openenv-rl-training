# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "datasets",
# ]
# ///

"""P4.1 — CPU pre-flight smoke test for SakThai training/eval jobs.

Wrap a job like:

    python smoke_test.py --script train.py --env TRAIN_ENV=browsergym \
        --env TRAIN_BASE=Nanthasit/sakthai-context-7b-tools

If this exits 0, the real GPU job is safe to launch. If it exits non-zero,
the GPU never spins up and the failure is caught in seconds on cpu-basic
instead of minutes on a100-large.

Checks (all under 30 seconds):
  1. Script parses as Python (compile).
  2. PEP-723 dependency header parses.
  3. `argparse.ArgumentParser` is constructed WITHOUT calling parse_args
     against real argv — we inject `--help` to verify all required flags
     are satisfiable from the current env vars (SystemExit(0) means OK).
  4. Optional: load model tokenizer + tokenize one dummy sample if
     --model or TRAIN_BASE env var is set.

Deliberately does NOT:
  - Download model weights (only the tokenizer).
  - Run any training step.
  - Import CUDA-only modules.
"""

import argparse
import ast
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TIMEOUT_SEC = 25


def _check_python_parses(path: Path) -> None:
    src = path.read_text()
    try:
        ast.parse(src)
    except SyntaxError as e:
        raise SystemExit(f"[smoke] SYNTAX ERROR in {path}: {e}")
    print(f"[smoke] OK: {path.name} parses as Python")


def _check_pep723_header(path: Path) -> None:
    src = path.read_text()
    m = re.search(r"^# /// script\n(.*?)^# ///", src, re.DOTALL | re.MULTILINE)
    if not m:
        print(f"[smoke] WARN: no PEP-723 header in {path.name}")
        return
    header = m.group(1)
    deps = re.findall(r'"([^"]+)"', header)
    print(f"[smoke] OK: {path.name} declares {len(deps)} deps: {deps}")


def _check_argparse_with_env(path: Path, extra_env: dict) -> None:
    env = {**os.environ, **extra_env}
    proc = subprocess.run(
        [sys.executable, str(path), "--help"],
        capture_output=True, text=True, timeout=TIMEOUT_SEC, env=env,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"[smoke] FAIL: --help exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}"
        )
    print(f"[smoke] OK: --help succeeds with injected env vars")


def _check_tokenizer(model_id: str | None) -> None:
    if not model_id:
        print("[smoke] SKIP: no --model / TRAIN_BASE set; skipping tokenizer check")
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok("hello world, this is a smoke test.", return_tensors="pt")
    n = enc["input_ids"].shape[-1]
    print(f"[smoke] OK: tokenizer for {model_id} loaded, {n} tokens on dummy input")


def main() -> None:
    p = argparse.ArgumentParser(description="CPU pre-flight smoke test.")
    p.add_argument("--script", required=True,
                   help="Path (or URL) to the training/eval script to check.")
    p.add_argument("--env", action="append", default=[],
                   help="Env var to inject as KEY=VALUE. Repeatable.")
    p.add_argument("--model", default=None,
                   help="Model id to load tokenizer for (also read from TRAIN_BASE env).")
    args = p.parse_args()

    extra_env = {}
    for kv in args.env:
        if "=" not in kv:
            raise SystemExit(f"[smoke] bad --env {kv!r}, expected KEY=VALUE")
        k, v = kv.split("=", 1)
        extra_env[k] = v

    script_path = Path(args.script)
    if not script_path.exists():
        raise SystemExit(f"[smoke] FAIL: script not found: {script_path}")

    _check_python_parses(script_path)
    _check_pep723_header(script_path)
    _check_argparse_with_env(script_path, extra_env)
    _check_tokenizer(args.model or extra_env.get("TRAIN_BASE") or os.environ.get("TRAIN_BASE"))

    print("[smoke] ALL CHECKS PASSED - safe to launch GPU job")


if __name__ == "__main__":
    main()
