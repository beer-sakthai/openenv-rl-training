# SakThai OpenEnv RL Training

End-to-end training, alignment, and evaluation pipeline for the **SakThai** agent model family (tool-calling and agentic reasoning models based on Qwen2.5).

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Models-SakThai%20Family-yellow)](https://huggingface.co/collections/Nanthasit/sakthai-model-family-6a64745450b12d421c1f9f02)
[![CI](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/verify-contracts.yml/badge.svg?branch=main)](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/verify-contracts.yml)
[![Tests](https://img.shields.io/badge/tests-160%20passed-brightgreen.svg)](tests)
[![Linter](https://img.shields.io/badge/linter-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Security Policy](https://img.shields.io/badge/security-policy-blue.svg)](SECURITY.md)
[![Known Gaps](https://img.shields.io/badge/known%20gaps-6%20tracked-blue)](docs/KNOWN_GAPS.yaml)

---

## What is this repository for?

This repository builds, aligns, and validates the machine learning models behind the SakThai agent ecosystem. It coordinates two training stages:

1. **Supervised Fine-Tuning (SFT)** (`sakthai-sft-training/`):
   - Fine-tunes Qwen2.5 (0.5B, 1.5B, 7B) using **QLoRA** on curated function-calling and multi-turn conversational datasets.
   - Implements automated multi-cycle dataset augmentation, safety/quality audits, contamination audits, and composition stats.
   - Produces the foundation tool-calling adapter: [`Nanthasit/sakthai-context-7b-tools`](https://huggingface.co/Nanthasit/sakthai-context-7b-tools).

2. **Reinforcement Learning via GRPO** (`openenv-custom-training/`, `openenv-multi-catalog-training/`):
   - Takes the 7B SFT adapter and applies **Group Relative Policy Optimization (GRPO)** using TRL and [Hugging Face OpenEnv](https://github.com/huggingface/OpenEnv).
   - Trains agents on interactive multi-turn tasks (such as **BrowserGym MiniWoB++** for web automation and 8-domain catalog environments).
   - Produces final RL-aligned checkpoints: [`Nanthasit/sakthai-context-7b-tools-grpo`](https://huggingface.co/Nanthasit/sakthai-context-7b-tools-grpo).

```text
[ Data Augmentation ] ──> [ SFT (QLoRA) ] ──> [ 7B Base Adapter ] ──> [ GRPO + OpenEnv ] ──> [ Final Checkpoint ]
                                                                                                    │
                                                                                                    ▼
                                                                     Deployed to: beer-sakthai/Sak-Family-Agent
```

> **Ecosystem Note**: This repository trains and evaluates models. The end-user application (with SQLite memory, MCP servers, and persona interfaces) lives in [`beer-sakthai/Sak-Family-Agent`](https://github.com/beer-sakthai/Sak-Family-Agent). The two repositories pin separate dependencies and share no runtime code.

---

## Published Models & Datasets

All assets are available on the Hugging Face Hub under [`Nanthasit`](https://huggingface.co/Nanthasit). Beyond the core RL tool-calling pipeline, the ecosystem encompasses **18 verified models** spanning Coding, Vision, Speech, 128k Context, and Embeddings (detailed in [`docs/MODELS.md`](docs/MODELS.md)).

### Core Tool-Calling & RL Matrix

| Model Tier | Base Model | Role & Primary Strength | Multi-Turn Agentic | GRPO Target? |
| :--- | :--- | :--- | :--- | :--- |
| [`sakthai-context-0.5b-tools`](https://huggingface.co/Nanthasit/sakthai-context-0.5b-tools) | Qwen2.5-0.5B | Fast tool routing; highest single-shot selection (**91.0%**) | 0/6 (Fails) | ❌ SFT-only |
| [`sakthai-context-1.5b-tools`](https://huggingface.co/Nanthasit/sakthai-context-1.5b-tools) | Qwen2.5-1.5B | Balanced model for consumer GPUs (Colab T4, 4–8 GB VRAM) | 0/6 (Fails) | ⚠️ Not targeted |
| [`sakthai-context-7b-tools`](https://huggingface.co/Nanthasit/sakthai-context-7b-tools) | Qwen2.5-7B | Flagship adapter; complex multi-step reasoning & tool calling | **3/6** (Passed) | ✅ **Primary target** |
| `sakthai-context-7b-tools-grpo` | 7B + GRPO | Production RL checkpoint aligned on OpenEnv tasks | In-loop eval | 🎯 **Final output** |

### Datasets

| Dataset | Type | Rows | Description |
| :--- | :--- | :--- | :--- |
| [`sakthai-combined-v12`](https://huggingface.co/datasets/Nanthasit/sakthai-combined-v12) | SFT Corpus | ~5,000 | Curated multi-turn instruction and tool-calling dataset. |
| [`sakthai-bench-v3`](https://huggingface.co/datasets/Nanthasit/sakthai-bench-v3) | Evaluation | 155 | Balanced function-calling benchmark harness across 15 capability categories. |

*Detailed benchmark tables live in [`sakthai-sft-training/README.md`](sakthai-sft-training/README.md) (SFT) and [`sakthai-agentic-eval-train/FINDINGS.md`](sakthai-agentic-eval-train/FINDINGS.md) (RL).*

---

## Repository Workspaces

This repository is organized into five **independent workspaces** (not an installable monolithic package):

| Workspace | Purpose | Key Scripts & Entrypoints | Tests |
| :--- | :--- | :--- | :--- |
| **`sakthai-sft-training/`** | SFT QLoRA trainers (0.5B/1.5B/7B), embedding model, 10-cycle augmentation, eval scripts, and ops guardrails. | `train_qwen.py`, `train-sakthai-embedding.py`, `scripts/eval/`, `scripts/ops/` | 37 tests (`tests/test_eval_cross_model.py`, `test_train_embedding.py`, `test_embedding_trainer.py`, `test_eval_scripts.py`, `test_ops_scripts.py`) |
| **`openenv-custom-training/`** | Custom OpenEnv environments, heuristic baselines, MCP task env, BrowserGym, and GRPO training runners. | `train.py`, `multi_env.py`, `evaluate_baselines.py`, `env_simple_task.py`, `agent_tools/` | 53 tests (`tests/test_*.py`) |
| **`openenv-multi-catalog-training/`** | Multi-environment GRPO runner across 8 OpenEnv catalog environments + A2A protocol agent. | `train_multi_env.py`, `run_servers.sh`, `a2a_agent/server.py`, `a2a_agent/client_smoke_test.py` | 25 tests (`tests/test_multi_catalog_env.py`, `test_a2a_agent.py`) |
| **`sakthai-agentic-eval-train/`** | Empirical agentic evaluation harness, PEFT benchmark runner, Colab notebooks, and findings ledger. | `eval_bench_peft.py`, `eval_hermes_env.py`, `scripts/benchmark_evaluation_manager_agent.py` | 30 tests (`tests/test_*.py`) |
| **`browsergym-space/`** | Dockerfile and Space configuration for a BrowserGym OpenEnv server. | `Dockerfile`, `app.py` | Validated via contract |
| **Repo Root & CI** | Root contract checks, CPU stub provider (`conftest.py`), and CI workflow definitions. | `verify_grpo_contract.py`, `test_browsergym_contract.py`, `conftest.py`, `.github/workflows/` | 15 contract tests |

---

## OpenEnv & TRL Contract Conventions

Every environment in this repo honours the TRL `environment_factory` contract:

- **0-Argument Initialization**: `__init__(self)` takes no required arguments. Configuration is read from module constants or environment variables.
- **Conversational Prompts**: `prompt` must be a list of chat message dicts (`[{"role": "user", "content": ...}]`), never a raw string.
- **Google-Style Docstrings with `Args:`**: Every public method other than `reset` is converted into an LLM tool. Parameters missing an `Args:` block cause `transformers.get_json_schema` to raise `DocstringParsingException`.
- **Post-Completion Guards**: Calling tools after episode termination (`self.done == True`) raises `ValueError` immediately.
- **Capped Steps**: All environments enforce step limits (`MAX_ATTEMPTS`, `STEP_LIMIT`) to ensure non-converging rollouts terminate cleanly with `0.0` reward.
- **Binary Outcome Rewards**: Rewards evaluate the final episode state, returning `1.0` (success) or `0.0` (failure).

---

## Local Verification & Testing (CPU)

All test suites and contract checks are fully runnable on a standard CPU machine without local GPUs, Hugging Face tokens, or running Docker containers. The root [`conftest.py`](conftest.py) provides lightweight stubs (`openenv`, `trl`, `a2a`) when training packages are absent.

### 1. Comprehensive Test Suite (160 Tests in Single Invocation)

Run all workspace test suites in one invocation (matching GitHub Actions CI):

```bash
uv run --with datasets --with pytest --with pydantic --with httpx --with anyio \
  pytest test_browsergym_contract.py \
         sakthai-agentic-eval-train/tests/ \
         openenv-custom-training/tests/ \
         openenv-multi-catalog-training/tests/ \
         sakthai-sft-training/tests/ -v
```

### 2. Standalone Workspace Suite Testing

Because [`conftest.py`](conftest.py) isolates stubs cleanly, any workspace test suite can also be executed independently:

```bash
# Test SFT data governance and cross-model eval tooling (37 tests)
uv run --with datasets --with pytest --with pydantic pytest sakthai-sft-training/tests/

# Test multi-catalog environment and A2A protocol agent (25 tests)
uv run --with datasets --with pytest --with pydantic --with httpx --with anyio pytest openenv-multi-catalog-training/tests/

# Test custom environments, baselines, and sandboxes (53 tests)
uv run --with datasets --with pytest --with pydantic pytest openenv-custom-training/tests/

# Test agentic evaluation pipeline and evaluation manager (30 tests)
uv run --with datasets --with pytest --with pydantic --with httpx --with anyio pytest sakthai-agentic-eval-train/tests/

# Test root contract suite (15 tests)
uv run --with datasets --with pytest --with pydantic pytest test_browsergym_contract.py
```

### 3. Static Linter Check

Ensure zero syntax errors, undefined names, redefinitions, and unused imports/variables:

```bash
uv run --with ruff ruff check .
```

### 4. OpenEnv RL Contract Verification

Drives `SimpleGuessEnv` through manual rollouts, verifies `environment_factory` structural compliance, checks reward calculations, and validates `GRPOConfig` schema compatibility:

```bash
uv run --with datasets python3 verify_grpo_contract.py
```

### 5. Baseline Heuristic Competition & GRPO Reward Variance

GRPO requires non-zero advantage within rollout groups. If all generations receive identical rewards, advantage is zero and gradients die (`frac_zero_std == 1.0`). This tool evaluates heuristic policies and Catch mechanics:

```bash
uv run --with datasets --with pydantic python3 openenv-custom-training/evaluate_baselines.py
```

### 6. Evaluation Manager Agent Benchmark

Deterministic rubric benchmark evaluating the Claude Code agent configuration for single-shot eval, agentic eval, cross-model comparison, and remote execution safety:

```bash
python3 sakthai-agentic-eval-train/scripts/benchmark_evaluation_manager_agent.py
```

Use `--json` for machine-readable output. The benchmark and its focused regression tests run automatically on every pull request and push to `main` through [`benchmark-evaluation-manager-agent.yml`](.github/workflows/benchmark-evaluation-manager-agent.yml). See the [agent configuration and extension guide](docs/evaluation-manager-agent.md) before adding a new evaluator or changing the rubric.

---

## Operational Tooling & Scripts

### Catalog Docker Servers (`run_servers.sh`)

Manages the 8 OpenEnv catalog environment containers (`echo`, `sudoku`, `coding`, `chat`, `atari`, `openspiel`, `repl`, `sumo` on ports `8001`–`8008`):

```bash
# Inspect the planned container launch commands without starting Docker
bash openenv-multi-catalog-training/run_servers.sh --dry-run

# Check status of running catalog containers
bash openenv-multi-catalog-training/run_servers.sh --status
```

### SFT Data Governance & Contamination Audit

```bash
# Audit benchmark prompts against training datasets for contamination
python3 sakthai-sft-training/scripts/eval/contamination_audit.py \
  --bench Nanthasit/sakthai-bench-v3 \
  --training Nanthasit/sakthai-combined-v10 Nanthasit/sakthai-combined-v12

# Classify dataset composition into simple, parallel, irrelevance, multi_turn
python3 sakthai-sft-training/scripts/eval/dataset_composition_stats.py Nanthasit/sakthai-combined-v12
```

### Preflight Smoke Testing & GPU Billing Downgrade Guard

```bash
# CPU preflight test before launching expensive GPU jobs
python3 sakthai-sft-training/scripts/ops/smoke_test.py --script train.py

# Guarded job launcher: automatically downgrades debug/test commands to cpu-basic
python3 sakthai-sft-training/scripts/ops/submit_job.py \
  --tag prod-train \
  --flavor a100-large \
  --script https://huggingface.co/.../train.py \
  --dry-run
```

---

## Remote GPU Training (Hugging Face Jobs)

When ready to train on GPU clusters, dispatch jobs via `hf jobs`:

```bash
# BrowserGym MiniWoB++ GRPO Training on A100 GPU
hf jobs uv run --detach --flavor a100-large --secrets HF_TOKEN --timeout 1h \
  -e TRAIN_MODE=lora16 \
  -e TRAIN_BASE=Nanthasit/sakthai-context-7b-tools \
  -e TRAIN_MAX_STEPS=150 \
  -e TRAIN_PUSH_TO=Nanthasit/sakthai-context-7b-tools-grpo \
  openenv-custom-training/train.py --env browsergym --browsergym-task click-button
```

*Interactive notebooks for free Colab T4 GPUs are available at [`sakthai-agentic-eval-train/sakthai_grpo_colab.ipynb`](sakthai-agentic-eval-train/sakthai_grpo_colab.ipynb) (GRPO training) and [`sakthai-agentic-eval-train/eval_sakthai_bench_colab.ipynb`](sakthai-agentic-eval-train/eval_sakthai_bench_colab.ipynb) (PEFT benchmark evaluation).*

---

## Security & Quality Assurance

This repository enforces multi-layered security controls and continuous quality gates across code, workflows, and execution environments:

- **Least-Privilege Token Permissions**: All GitHub Actions workflows declare top-level read-only permissions (`permissions: contents: read`) with step-level least privilege to prevent token leakage or unauthorized repository mutations.
- **StepSecurity Harden-Runner**: CI jobs run on egress-filtered and monitored runners (`step-security/harden-runner`) with policy store verification.
- **Execution Sandbox Confinement**: The terminal sandbox environment (`openenv-custom-training/agent_tools/server/sandbox_env.py`) strictly confines filesystem mutations to per-episode temporary scratch workspaces (`_confine()`), prevents directory traversal / symlink escapes, restricts environment variables to safe defaults (`PATH`, `HOME`, `LANG`), and enforces strict timeouts (`CMD_TIMEOUT_S = 10`) and output caps.
- **Safe Network & URL Validation**: Remote evaluators enforce strict URL scheme whitelist validation (`https://` and `http://` only), preventing unexpected schemes (`file://`, `ftp://`).
- **Static Analysis & Security Audits**: Checked via Ruff (`E9`, `F821`, `F811`, `F401`, `F841`), Bandit (CWE scanning), CodeQL semantic analysis, OSSAR, and Bandit secret scanning.
- **Deterministic Evaluation Rubrics**: The evaluation manager agent benchmark enforces reproducible evaluations and strictly forbids reporting unverified GPU operations or unrun HF Jobs.
- **Detailed Security Policy**: Full token hygiene checklists, vulnerability reporting procedures, and emergency response guides are documented in [`SECURITY.md`](SECURITY.md).

---

## Project Status & Governance

- **CI Workflow**: [`.github/workflows/verify-contracts.yml`](.github/workflows/verify-contracts.yml) runs ruff, contract verifications, baseline competitions, and all 160 pytest tests across 5 workspace suites on every push and PR.
- **Empirical Findings**: Benchmark evaluations and capacity observations are permanently recorded in [`sakthai-agentic-eval-train/FINDINGS.md`](sakthai-agentic-eval-train/FINDINGS.md).
- **Known Gaps**: Tracked issues, workarounds, and blockers are listed in [`docs/KNOWN_GAPS.yaml`](docs/KNOWN_GAPS.yaml).
- **Changelog**: Release history and completed improvements are tracked in [`CHANGELOG.md`](CHANGELOG.md).
- **Security Policy**: Vulnerability disclosure and operational token checklists live in [`SECURITY.md`](SECURITY.md).

---

## License

[Apache-2.0](LICENSE)
