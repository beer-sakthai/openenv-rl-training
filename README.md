# OpenEnv + TRL GRPO training — SakThai family

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

End-to-end training + evaluation pipeline for the SakThai model family — both the
supervised half (QLoRA on Qwen2.5 for tool-calling) and the reinforcement-learning half
(GRPO with [Hugging Face OpenEnv](https://github.com/huggingface/OpenEnv) and
[TRL](https://huggingface.co/docs/trl)'s `GRPOTrainer` `environment_factory`).

Consolidated on 2026-08-21 from two repos (this one and the retired `beer-sakthai/SakThai-Training`,
deleted; history preserved in the local archive).

## Layout

| Directory | What it is |
|-----------|------------|
| [`sakthai-sft-training/`](sakthai-sft-training/) | **SFT half.** QLoRA training scripts (0.5B / 1.5B / 7B), 10-cycle data-augmentation loop (`cycle-100-v*.py`), cross-model / MCP-Bench / lighteval evaluators, `sakthai-cycle-bench/` (155-row BFCL harness), ops tooling (`scripts/ops/`, `scripts/eval/`), self-contained SFT Colab notebook. |
| [`openenv-custom-training/`](openenv-custom-training/) | **RL — custom environments.** Tier A (inline plain-Python logic, no server), Tier B (sandboxed OpenEnv server), and **BrowserGym MiniWoB++** (`--env browsergym`), plus `train.py` (single-env) and `multi_env.py` (TRL-native dict-form multi-env) runners. Default base: `Nanthasit/sakthai-context-7b-tools`. |
| [`openenv-multi-catalog-training/`](openenv-multi-catalog-training/) | **RL — catalog run.** Trains one small model across all 8 `openenv/*` catalog environments (echo, sudoku, coding, chat, atari, openspiel, repl, sumo) in a single GRPO run via the multi-environment pattern. |
| [`sakthai-agentic-eval-train/`](sakthai-agentic-eval-train/) | **The eval + train pipeline that actually ran.** As-run HF Jobs scripts + `sakthai_grpo_colab.ipynb` + `FINDINGS.md` (the durable empirical record — read this before re-litigating model choice). |
| [`browsergym-space/`](browsergym-space/) | **BrowserGym OpenEnv Server** — deployed live on Hugging Face Spaces at [`Nanthasit/browsergym-env`](https://huggingface.co/spaces/Nanthasit/browsergym-env) (`https://nanthasit-browsergym-env.hf.space`). |

## End-to-end pipeline

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
`sakthai-agentic-eval-train/FINDINGS.md` is the single source of truth for what has
actually worked.

## Status — Verified Contract & BrowserGym Integrated (2026-08-03)

- **BrowserGym OpenEnv Server (`browsergym-space/`)**: Deployed live on HF Spaces at [`Nanthasit/browsergym-env`](https://huggingface.co/spaces/Nanthasit/browsergym-env). Serves Gymnasium-compatible web navigation tasks (MiniWoB++ click, form-fill, navigation).
- **`train.py --env browsergym`**: BrowserGym environment factory & reward extraction integrated in `openenv-custom-training/train.py`.
- **Local Verification Contract (`verify_grpo_contract.py` & `test_browsergym_contract.py`)**: Run `pytest test_browsergym_contract.py` and `python3 verify_grpo_contract.py` to verify reward math and contracts on CPU before GPU runs. CI wired via `.github/workflows/verify-contracts.yml` (2026-08-21).
- **Target Model (`Nanthasit/sakthai-context-7b-tools`)**: Primary viable target for GRPO RL training (showing non-zero reward variance and `grad_norm` 0.25–0.49).

## Launching GRPO GPU Training Jobs on HF Jobs

```bash
# BrowserGym MiniWoB++ RL Training on A100 GPU
hf jobs uv run --detach --flavor a100-large --secrets HF_TOKEN --timeout 1h \
  -e TRAIN_MODE=lora16 -e TRAIN_BASE=Nanthasit/sakthai-context-7b-tools \
  -e TRAIN_MAX_STEPS=150 -e TRAIN_EPISODES=8 -e TRAIN_MAX_COMPLETION=1024 \
  -e TRAIN_PUSH_TO=Nanthasit/sakthai-context-7b-tools-grpo \
  openenv-custom-training/train.py --env browsergym --browsergym-task click-button
```

For the SFT half, see [`sakthai-sft-training/README.md`](sakthai-sft-training/README.md).

## CI

- `.github/workflows/verify-contracts.yml` — runs the two CPU contract tests on every PR. Free.
- `.github/workflows/{train,eval,lighteval,mcp-bench,monitor}.yml` — dispatch `hf jobs uv run` to HF Jobs. Need `HF_TOKEN` set as a repo secret + a paid HF Jobs plan. `train.yml` is `workflow_dispatch`-only to avoid unintentional GPU spend.

## License

[Apache-2.0](LICENSE), consistent with the SakThai family's published artifacts.
