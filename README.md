# 🚀 OpenEnv + TRL GRPO training — SakThai family

<!-- 📊 Status bar — CI, security, licence, hygiene, Hub assets -->

[![✅ Verify Contracts](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/verify-contracts.yml/badge.svg?branch=main)](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/verify-contracts.yml)
[![🛡️ CodeQL](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/codeql.yml)
[![🔎 OSSAR](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/ossar.yml/badge.svg?branch=main)](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/ossar.yml)
[![📦 Dependency Review](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/dependency-review.yml/badge.svg?branch=main)](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/dependency-review.yml)
[![🧹 Stale](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/stale.yml/badge.svg?branch=main)](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/stale.yml)
[![🔄 Auto Merge](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/auto-merge.yml/badge.svg?branch=main)](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/auto-merge.yml)
[![⬆️ Auto Update PRs](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/auto-update-prs.yml/badge.svg?branch=main)](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/auto-update-prs.yml)
[![📊 Eval](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/eval.yml/badge.svg?branch=main)](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/eval.yml)
[![⚡ Lighteval](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/lighteval.yml/badge.svg?branch=main)](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/lighteval.yml)
[![✋ Manual](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/manual.yml/badge.svg?branch=main)](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/manual.yml)
[![⚖️ MCP Bench](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/mcp-bench.yml/badge.svg?branch=main)](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/mcp-bench.yml)
[![👀 Monitor](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/monitor.yml/badge.svg?branch=main)](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/monitor.yml)
[![📝 Summary](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/summary.yml/badge.svg?branch=main)](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/summary.yml)
[![🚂 Train](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/train.yml/badge.svg?branch=main)](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/train.yml)

[![📜 License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![🐍 Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![🤗 Hugging Face](https://img.shields.io/badge/🤗-SakThai%20Family-yellow)](https://huggingface.co/collections/Nanthasit/sakthai-model-family-6a64745450b12d421c1f9f02)
[![🌿 Branch policy: main only](https://img.shields.io/badge/branch--policy-main%20only-brightgreen)](docs/branch-cleanup-2026-08-22.md)
[![🔁 Last cleanup](https://img.shields.io/badge/last%20cleanup-2026--08--22-informational)](docs/branch-cleanup-2026-08-22.md)
[![💬 Issues](https://img.shields.io/github/issues/beer-sakthai/openenv-rl-training?label=issues&logo=github)](https://github.com/beer-sakthai/openenv-rl-training/issues)
[![🔀 Pull Requests](https://img.shields.io/github/issues-pr/beer-sakthai/openenv-rl-training?label=PRs&logo=github)](https://github.com/beer-sakthai/openenv-rl-training/pulls)
[![⭐ Stars](https://img.shields.io/github/stars/beer-sakthai/openenv-rl-training?style=flat&logo=github)](https://github.com/beer-sakthai/openenv-rl-training/stargazers)
[![📅 Last commit](https://img.shields.io/github/last-commit/beer-sakthai/openenv-rl-training/main?logo=git&logoColor=white)](https://github.com/beer-sakthai/openenv-rl-training/commits/main)

### 🤗 Model & dataset badges

[![🧠 sakthai-context-0.5b-tools](https://img.shields.io/badge/🤗_model-sakthai--context--0.5b--tools-orange)](https://huggingface.co/Nanthasit/sakthai-context-0.5b-tools)
[![🧠 sakthai-context-1.5b-tools](https://img.shields.io/badge/🤗_model-sakthai--context--1.5b--tools-orange)](https://huggingface.co/Nanthasit/sakthai-context-1.5b-tools)
[![🧠 sakthai-context-7b-tools](https://img.shields.io/badge/🤗_model-sakthai--context--7b--tools-orange)](https://huggingface.co/Nanthasit/sakthai-context-7b-tools)
[![🌐 browsergym-env](https://img.shields.io/badge/🤗_space-browsergym--env-blue)](https://huggingface.co/spaces/Nanthasit/browsergym-env)
[![📊 sakthai-combined-v12](https://img.shields.io/badge/🤗_dataset-sakthai--combined--v12-green)](https://huggingface.co/datasets/Nanthasit/sakthai-combined-v12)
[![📊 sakthai-bench-v3](https://img.shields.io/badge/🤗_dataset-sakthai--bench--v3-green)](https://huggingface.co/datasets/Nanthasit/sakthai-bench-v3)

---

End-to-end training + evaluation pipeline for the **SakThai model family** — both the
supervised half (QLoRA on Qwen2.5 for tool-calling) and the reinforcement-learning half
(GRPO with [🤗 OpenEnv](https://github.com/huggingface/OpenEnv) and
[TRL](https://huggingface.co/docs/trl)'s `GRPOTrainer` `environment_factory`).

📚 Consolidated on **2026-08-21** from two repos (this one and the retired
`beer-sakthai/SakThai-Training`, deleted; history preserved in the local archive).
🧹 Branch-cleanup pass **2026-08-22** brought the repo to a single-branch (`main` only)
state — see [`docs/branch-cleanup-2026-08-22.md`](docs/branch-cleanup-2026-08-22.md).

## 📂 Layout

| 📁 Directory | 📝 What it is |
|-----------|------------|
| 🎓 [`sakthai-sft-training/`](sakthai-sft-training/) | **SFT half.** QLoRA training scripts (0.5B / 1.5B / 7B), 10-cycle data-augmentation loop (`cycle-100-v*.py`), cross-model / MCP-Bench / lighteval evaluators, `sakthai-cycle-bench/` (155-row BFCL harness), ops tooling (`scripts/ops/`, `scripts/eval/`), self-contained SFT Colab notebook. |
| 🎯 [`openenv-custom-training/`](openenv-custom-training/) | **RL — custom environments.** Tier A (inline plain-Python logic, no server), Tier B (sandboxed OpenEnv server), and **BrowserGym MiniWoB++** (`--env browsergym`), plus `train.py` (single-env) and `multi_env.py` (TRL-native dict-form multi-env) runners. Default base: `Nanthasit/sakthai-context-7b-tools`. |
| 🎮 [`openenv-multi-catalog-training/`](openenv-multi-catalog-training/) | **RL — catalog run.** Trains one small model across all 8 `openenv/*` catalog environments (echo, sudoku, coding, chat, atari, openspiel, repl, sumo) in a single GRPO run via the multi-environment pattern. |
| 🧪 [`sakthai-agentic-eval-train/`](sakthai-agentic-eval-train/) | **The eval + train pipeline that actually ran.** As-run HF Jobs scripts + `sakthai_grpo_colab.ipynb` + `FINDINGS.md` (the durable empirical record — read this before re-litigating model choice). |
| 🌐 [`browsergym-space/`](browsergym-space/) | **BrowserGym OpenEnv Server** — deployed live on Hugging Face Spaces at [`Nanthasit/browsergym-env`](https://huggingface.co/spaces/Nanthasit/browsergym-env) (`https://nanthasit-browsergym-env.hf.space`). |
| 📖 [`docs/`](docs/) | Repo audit & policy docs — 2026-07-30 Hub audit, `SECURITY.md`, and the 2026-08-22 branch-cleanup record. |
| ⚙️ [`.github/workflows/`](.github/workflows/) | 14 GitHub Actions workflows — CI (`verify-contracts`), security (`codeql`, `ossar`, `dependency-review`), hygiene (`stale`, `summary`), and 5 HF-Jobs dispatchers (`train`, `eval`, `lighteval`, `mcp-bench`, `monitor`). |
| 🤖 [`.opencode/`](.opencode/) | 25 slash-command specs + 35 workflow skills — a path-agnostic prompt library, no code depends on it. |

## 🔄 End-to-end pipeline

```
              sakthai-sft-training/                sakthai-agentic-eval-train/
              (QLoRA + augmentation)               (bench + agentic + GRPO pilot)
                     |                                        ^
                     v                                        |
              Nanthasit/sakthai-context-7b-tools  ---GRPO---> Nanthasit/sakthai-context-7b-tools-grpo
                     ^                                        |
                     |                                        v
              cycle-100-v*.py  <---(FINDINGS.md drives next round)---  openenv-{custom,multi-catalog}-training/
```

The SFT half produces the base LoRA. The RL half loads that base, runs GRPO on OpenEnv
environments (custom or catalog), and pushes a merged bf16 checkpoint back to the Hub.
📄 [`sakthai-agentic-eval-train/FINDINGS.md`](sakthai-agentic-eval-train/FINDINGS.md) is
the single source of truth for what has actually worked.

## 📊 Status — 2026-08-22

### 🟢 Repository hygiene

- 🌿 **Branch policy: `main` only.** The 2026-08-22 cleanup pass reduced the repo from
  `main` + 34 non-main remote branches to `main` alone. Audit trail:
  [`docs/branch-cleanup-2026-08-22.md`](docs/branch-cleanup-2026-08-22.md).
- 🔀 **Zero open PRs.** All 18 in-flight PRs were resolved: 12 squash-merged
  (docs/tests/bugfixes), 7 closed unmerged (duplicates or `CLAUDE.md`-forbidden
  workflow reintroductions).
- 🧨 **22 orphan branch refs deleted** via a one-shot `workflow_dispatch` job
  (`git push --delete` is blocked at the session's egress proxy).

### 🟢 Verified contracts & CI

- ✅ **`verify_grpo_contract.py`** — CPU-only reward-function + `GRPOConfig`
  compatibility check. Passes on `main`.
- ✅ **`test_browsergym_contract.py`** — 3 tests, mocks `trl`, requires `datasets`.
  Run with `uv run --with datasets --with pytest pytest test_browsergym_contract.py`.
- ⚙️ **`.github/workflows/verify-contracts.yml`** — runs both on every PR & push
  to `main`. Free; the only workflow that runs inside GH Actions itself.
- 🛡️ **`.github/workflows/codeql.yml`** & **`ossar.yml`** — advanced code-scanning
  for actions & Python; weekly + on-PR.

### 🟢 Deployed assets

- 🌐 **BrowserGym OpenEnv Server (`browsergym-space/`)** — live at
  [`Nanthasit/browsergym-env`](https://huggingface.co/spaces/Nanthasit/browsergym-env).
  Serves Gymnasium-compatible web-navigation tasks (MiniWoB++ click, form-fill,
  navigation).
- 🎯 **`train.py --env browsergym`** — BrowserGym environment factory & reward
  extraction integrated in `openenv-custom-training/train.py`.
- 🧠 **Target model (`Nanthasit/sakthai-context-7b-tools`)** — the primary viable
  GRPO target (non-zero reward variance, `grad_norm` 0.25–0.49). See
  [`FINDINGS.md`](sakthai-agentic-eval-train/FINDINGS.md) for why.

### 🟡 Known open items (see `CLAUDE.md`)

- 💰 **HF Jobs currently returns `402 Payment Required`** on this account —
  the five HF-Jobs workflows will fail until this is resolved and `HF_TOKEN`
  is added as a repo secret. `verify-contracts.yml` runs regardless.
- 🧪 **`coding_env` task placeholder** in both `train_multi_env.py` and
  `a2a_agent/` — substring-check for `print(17 * 23)`.
- 🐳 **Catalog Docker image tags in `run_servers.sh`** — none verified live.
- 🕸️ **`a2a_agent/` never executed** — `TaskUpdater` method names were written
  from the published SDK pattern, not a live install.
- ⏱️ **7B GRPO proof-of-signal run was 40 steps** — long enough to show a gradient
  exists, not long enough to improve the model. Real run needs hundreds of steps.

## 🚀 Launching GRPO GPU Training Jobs on HF Jobs

```bash
# 🌐 BrowserGym MiniWoB++ RL Training on A100 GPU
hf jobs uv run --detach --flavor a100-large --secrets HF_TOKEN --timeout 1h \
  -e TRAIN_MODE=lora16 -e TRAIN_BASE=Nanthasit/sakthai-context-7b-tools \
  -e TRAIN_MAX_STEPS=150 -e TRAIN_EPISODES=8 -e TRAIN_MAX_COMPLETION=1024 \
  -e TRAIN_PUSH_TO=Nanthasit/sakthai-context-7b-tools-grpo \
  openenv-custom-training/train.py --env browsergym --browsergym-task click-button
```

```bash
# 📊 Agentic eval on L4 GPU
hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \
  --env SAK_MODELS=Nanthasit/sakthai-context-7b-tools \
  sakthai-agentic-eval-train/scripts/eval_hermes_env.py
```

For the SFT half, see 📄 [`sakthai-sft-training/README.md`](sakthai-sft-training/README.md).

## ⚡ Running things locally (what actually works here)

```bash
# 🟢 Passes; skips the TRL class-instantiation section when trl is absent
python3 verify_grpo_contract.py

# 🟢 3 passed; mocks trl, needs datasets
uv run --with datasets --with pytest pytest test_browsergym_contract.py
```

Everything else needs a GPU box — this checkout has no `torch`/`trl`/`datasets`
installed by design; do **not** attempt to run training here.

## ⚙️ CI

| 🔧 Workflow | 🎯 Purpose | 🏃 When it runs | 💵 Cost |
|---|---|---|---|
| [`verify-contracts.yml`](.github/workflows/verify-contracts.yml) | Both CPU contract tests | Every PR + push to `main` | Free |
| [`codeql.yml`](.github/workflows/codeql.yml) | Static analysis (Python + Actions) | PR + weekly cron | Free |
| [`ossar.yml`](.github/workflows/ossar.yml) | Windows-based security scanners | Weekly cron | Free |
| [`dependency-review.yml`](.github/workflows/dependency-review.yml) | PR-only vuln/license diff review | Every PR | Free |
| [`stale.yml`](.github/workflows/stale.yml) | Issue/PR staleness reaper | Daily cron | Free |
| [`summary.yml`](.github/workflows/summary.yml) | LLM-generated PR summary | Every PR | Free |
| [`auto-merge.yml`](.github/workflows/auto-merge.yml) | Label-gated auto-merge (`automerge` label) | Label event | Free |
| [`auto-update-prs.yml`](.github/workflows/auto-update-prs.yml) | Auto-updates open PR branches when `main` moves | Push to `main` | Free |
| [`train.yml`](.github/workflows/train.yml) | Dispatch `hf jobs uv run` (GRPO) | `workflow_dispatch` | 💰 HF Jobs |
| [`eval.yml`](.github/workflows/eval.yml) | Dispatch eval to HF Jobs | Weekly cron / dispatch | 💰 HF Jobs |
| [`lighteval.yml`](.github/workflows/lighteval.yml) | Dispatch lighteval to HF Jobs | Weekly cron / dispatch | 💰 HF Jobs |
| [`mcp-bench.yml`](.github/workflows/mcp-bench.yml) | Dispatch MCP-Bench to HF Jobs | `workflow_dispatch` | 💰 HF Jobs |
| [`monitor.yml`](.github/workflows/monitor.yml) | Weekly HF Hub monitor | Weekly cron / dispatch | 💰 HF Jobs |

The HF-Jobs workflows require `HF_TOKEN` as a repo secret + a paid HF Jobs plan.
`train.yml` is `workflow_dispatch`-only to avoid unintentional GPU spend.

## 🔗 Related repositories

Three repos under [`beer-sakthai`](https://github.com/beer-sakthai) make up the SakThai
family. This one owns the **training and evaluation** pipeline; it ships no agent runtime.

| Repository | What it is | How it connects here |
|---|---|---|
| [`openenv-rl-training`](https://github.com/beer-sakthai/openenv-rl-training) | **This repo.** SFT (QLoRA on Qwen2.5 for tool-calling), GRPO over OpenEnv environments via TRL's `environment_factory`, the agentic-eval harness, and [`sakthai-agentic-eval-train/FINDINGS.md`](sakthai-agentic-eval-train/FINDINGS.md). | — |
| [`Sak-Family-Agent`](https://github.com/beer-sakthai/Sak-Family-Agent) | The runtime that consumes what this repo produces: the `sakthai` package, six personas, a persistent SQLite memory store, an MCP stdio server, and a web API. | Its `training/sakthai-7b-lora/train.py` pushes [`Nanthasit/sakthai-context-7b-tools`](https://huggingface.co/Nanthasit/sakthai-context-7b-tools) — the adapter this repo GRPO-trains further, and the one `FINDINGS.md` identifies as the only viable GRPO target in the family. The two repos share **no code** and pin deliberately incompatible dependency sets; keep them separate. |
| [`codeql-action`](https://github.com/beer-sakthai/codeql-action) | A fork of [`github/codeql-action`](https://github.com/github/codeql-action) carrying local dependency-advisory remediation against the action's own dev-dependency tree. | [`codeql.yml`](.github/workflows/codeql.yml) here pins **upstream** `github/codeql-action`, not the fork. |

Shared Hub assets — models, datasets, and the BrowserGym Space — live under
[`Nanthasit`](https://huggingface.co/Nanthasit); the badges at the top of this README
link the ones this repo produces or consumes.

## 📜 License

[Apache-2.0](LICENSE), consistent with the SakThai family's published artifacts.
