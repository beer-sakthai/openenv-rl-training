# OpenEnv + TRL GRPO training — SakThai family

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Reinforcement-learning (GRPO) fine-tuning of SakThai-family models with
[Hugging Face OpenEnv](https://github.com/huggingface/OpenEnv) (sandboxed,
Gymnasium-style environments) and [TRL](https://huggingface.co/docs/trl)'s
`GRPOTrainer` `environment_factory` (multi-turn tool-calling RL loop).

Companion workspaces:

| Directory | What it is |
|-----------|------------|
| [`openenv-custom-training/`](openenv-custom-training/) | **Custom environments** — Tier A (inline plain-Python logic, no server), Tier B (sandboxed OpenEnv server), and **BrowserGym MiniWoB++** (`--env browsergym`), plus `train.py` (single-env) and `multi_env.py` (TRL-native dict-form multi-env) runners. Default base: `Nanthasit/sakthai-context-7b-tools`. |
| [`openenv-multi-catalog-training/`](openenv-multi-catalog-training/) | **Catalog run** — trains one small model across all 8 `openenv/*` catalog environments (echo, sudoku, coding, chat, atari, openspiel, repl, sumo) in a single GRPO run via the multi-environment pattern. |
| [`browsergym-space/`](browsergym-space/) | **BrowserGym OpenEnv Server** — Deployed live on Hugging Face Spaces at [`Nanthasit/browsergym-env`](https://huggingface.co/spaces/Nanthasit/browsergym-env) (`https://nanthasit-browsergym-env.hf.space`). |

## Status — Verified Contract & BrowserGym Integrated (2026-08-03)

Both workspaces are updated with verified GRPO training contracts:

- **BrowserGym OpenEnv Server (`browsergym-space/`)**: Deployed live on HF Spaces at [`Nanthasit/browsergym-env`](https://huggingface.co/spaces/Nanthasit/browsergym-env). Serves Gymnasium-compatible web navigation tasks (MiniWoB++ click, form-fill, navigation).
- **`train.py --env browsergym`**: Integrated BrowserGym environment factory & reward extraction in `openenv-custom-training/train.py`.
- **Local Verification Contract (`verify_grpo_contract.py` & `test_browsergym_contract.py`)**: Run `pytest test_browsergym_contract.py` and `python3 verify_grpo_contract.py` to verify reward math and contracts on CPU before GPU runs.
- **Target Model (`Nanthasit/sakthai-context-7b-tools`)**: Primary viable target for GRPO RL training (showing non-zero reward variance and `grad_norm` 0.25–0.49).

### Launching GRPO GPU Training Jobs on HF Jobs

```bash
# BrowserGym MiniWoB++ RL Training on A100 GPU
hf jobs uv run --detach --flavor a100-large --secrets HF_TOKEN --timeout 1h \
  -e TRAIN_MODE=lora16 -e TRAIN_BASE=Nanthasit/sakthai-context-7b-tools \
  -e TRAIN_MAX_STEPS=150 -e TRAIN_EPISODES=8 -e TRAIN_MAX_COMPLETION=1024 \
  -e TRAIN_PUSH_TO=Nanthasit/sakthai-context-7b-tools-grpo \
  train.py --env browsergym --browsergym-task click-button
```

## License

[Apache-2.0](LICENSE), consistent with the SakThai family's published artifacts.
