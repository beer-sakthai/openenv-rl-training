#!/bin/bash
# SessionStart hook — install the minimal deps the two CPU contract tests
# (verify_grpo_contract.py, test_browsergym_contract.py) need to run.
#
# The repo is not a single installable package by design (see CLAUDE.md):
# each subdirectory owns its own deps and the heavy stack (torch, trl, vllm,
# ~GB) lives on a GPU box, not here. This hook installs only what's needed
# to make the local CPU tests green.
set -euo pipefail

# Only fire in Claude Code on the web (remote sessions). Locally you already
# have your env set up how you want it.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo "[session-start] setting up openenv-rl-training CPU test deps..."

# uv — the canonical invocation for test_browsergym_contract.py is
#   uv run --with datasets --with pytest pytest test_browsergym_contract.py
# and every PEP 723 script under sakthai-sft-training/ / openenv-*/ mirrors
# HF Jobs' `hf jobs uv run` locally as `uv run`.
python3 -m pip install --user --quiet uv || true

# pytest + datasets — for direct pytest invocation:
#   - verify_grpo_contract.py mocks trl (nothing extra needed)
#   - test_browsergym_contract.py mocks trl but imports datasets for real
python3 -m pip install --user --quiet pytest datasets

echo "[session-start] done. To run the contract tests:"
echo "  python3 verify_grpo_contract.py"
echo "  uv run --with datasets --with pytest pytest test_browsergym_contract.py"
