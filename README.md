# SakThai OpenEnv RL Training

End-to-end training and evaluation pipeline for the **SakThai** agent model family (tool-calling and agentic reasoning models based on Qwen2.5).

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Models-SakThai%20Family-yellow)](https://huggingface.co/collections/Nanthasit/sakthai-model-family-6a64745450b12d421c1f9f02)
[![CI](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/verify-contracts.yml/badge.svg?branch=main)](https://github.com/beer-sakthai/openenv-rl-training/actions/workflows/verify-contracts.yml)
[![Known Gaps](https://img.shields.io/badge/known%20gaps-10%20tracked-blue)](docs/KNOWN_GAPS.yaml)

---

## What is this repository for?

This repository builds and aligns the machine learning models behind the SakThai agent ecosystem. It coordinates two training stages:

1. **Supervised Fine-Tuning (SFT)** (`sakthai-sft-training/`):
   - Fine-tunes Qwen2.5 (0.5B, 1.5B, 7B) using **QLoRA** on curated function-calling and multi-turn conversational datasets.
   - Produces the foundation tool-calling adapter: [`Nanthasit/sakthai-context-7b-tools`](https://huggingface.co/Nanthasit/sakthai-context-7b-tools).

2. **Reinforcement Learning via GRPO** (`openenv-custom-training/`, `openenv-multi-catalog-training/`):
   - Takes the 7B SFT adapter and applies **Group Relative Policy Optimization (GRPO)** using TRL and [Hugging Face OpenEnv](https://github.com/huggingface/OpenEnv).
   - Trains agents on interactive multi-turn tasks (such as **BrowserGym MiniWoB++** for web automation and multi-domain catalog tasks).
   - Produces final RL-aligned checkpoints: [`Nanthasit/sakthai-context-7b-tools-grpo`](https://huggingface.co/Nanthasit/sakthai-context-7b-tools-grpo).

```text
[ Data Augmentation ] ──> [ SFT (QLoRA) ] ──> [ 7B Base Adapter ] ──> [ GRPO + OpenEnv ] ──> [ Final Checkpoint ]
                                                                                                    │
                                                                                                    ▼
                                                                     Deployed to: beer-sakthai/Sak-Family-Agent
```

> **Ecosystem Note**: This repository only trains and evaluates models. The end-user application (with SQLite memory, MCP servers, and persona interfaces) lives in [`beer-sakthai/Sak-Family-Agent`](https://github.com/beer-sakthai/Sak-Family-Agent).

---

## Published Models & Datasets

All assets are available on the Hugging Face Hub under [`Nanthasit`](https://huggingface.co/Nanthasit). Beyond the core RL tool-calling pipeline below, the ecosystem encompasses **13+ specialized models** spanning Coding, Vision, Speech, 128k Context, and Embeddings (detailed in [`docs/MODELS.md`](docs/MODELS.md)).

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
| [`sakthai-bench-v3`](https://huggingface.co/datasets/Nanthasit/sakthai-bench-v3) | Evaluation | 155 | Balanced function-calling benchmark harness. |

*The complete 13-model ecosystem catalog (including Vision, Coder, Plus, 128k Context, and TTS) is documented in [`docs/MODELS.md`](docs/MODELS.md).*

---

## Repository Structure

Each subdirectory is an independent workspace:

- **`sakthai-sft-training/`** — SFT training scripts, 10-cycle data augmentation loop (`cycle-100-v*.py`), and benchmark evaluators.
- **`openenv-custom-training/`** — GRPO training runners (`train.py`, `multi_env.py`) supporting BrowserGym MiniWoB++ and custom sandboxed environments.
- **`openenv-multi-catalog-training/`** — Multi-environment GRPO runner across 8 OpenEnv catalog environments.
- **`sakthai-agentic-eval-train/`** — Empirical evaluations, Colab notebooks, and [`FINDINGS.md`](sakthai-agentic-eval-train/FINDINGS.md) (empirical record).
- **`browsergym-space/`** — Dockerfile and Space configuration for the BrowserGym OpenEnv server.
- **`docs/`** — Architecture records, security policies, and the tracked open gap index ([`KNOWN_GAPS.yaml`](docs/KNOWN_GAPS.yaml)).
- **`.github/workflows/`** — Free CI test runners (`verify-contracts.yml`, `codeql.yml`) and Hugging Face Jobs dispatchers.

---

## Quickstart

### Local Verification (CPU)

Run linting and contract test suites locally:

```bash
# 1. Bug-only lint check (syntax, undefined names, redefinitions)
uv run --with ruff ruff check .

# 2. Verify GRPO environment contracts
python3 verify_grpo_contract.py

# 3. Run all 38 contract tests
uv run --with datasets --with pytest --with pydantic --with httpx \
  pytest test_browsergym_contract.py \
         sakthai-agentic-eval-train/tests/ \
         openenv-custom-training/tests/
```

### Remote GPU Training (Hugging Face Jobs)

Dispatch training runs directly to GPU clusters via `hf jobs`:

```bash
# BrowserGym MiniWoB++ GRPO Training on A100 GPU
hf jobs uv run --detach --flavor a100-large --secrets HF_TOKEN --timeout 1h \
  -e TRAIN_MODE=lora16 \
  -e TRAIN_BASE=Nanthasit/sakthai-context-7b-tools \
  -e TRAIN_MAX_STEPS=150 \
  -e TRAIN_PUSH_TO=Nanthasit/sakthai-context-7b-tools-grpo \
  openenv-custom-training/train.py --env browsergym --browsergym-task click-button
```

*For interactive experimentation, use the self-contained Colab notebook: [`sakthai-agentic-eval-train/sakthai_grpo_colab.ipynb`](sakthai-agentic-eval-train/sakthai_grpo_colab.ipynb).*

---

## CI & Project Status

- **CI Status**: Contract tests, CodeQL, and PR auto-updates run on every push to `main`.
- **Known Items**: Tracked issues, workarounds, and blockers are listed in [`docs/KNOWN_GAPS.yaml`](docs/KNOWN_GAPS.yaml).
- **Changelog**: Historical releases and completed PRs are tracked in [`CHANGELOG.md`](CHANGELOG.md).

---

## License

[Apache-2.0](LICENSE)
