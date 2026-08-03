# OpenEnv + TRL GRPO training — SakThai family

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Reinforcement-learning (GRPO) fine-tuning of SakThai-family models with
[Hugging Face OpenEnv](https://github.com/huggingface/OpenEnv) (sandboxed,
Gymnasium-style environments) and [TRL](https://huggingface.co/docs/trl)'s
`GRPOTrainer` `environment_factory` (multi-turn tool-calling RL loop).

Two companion workspaces:

| Directory | What it is |
|-----------|------------|
| [`openenv-custom-training/`](openenv-custom-training/) | **Custom environments** — Tier A (inline plain-Python logic, no server) and Tier B (sandboxed OpenEnv server for agent tool-use tasks), plus `train.py` (single-env) and `multi_env.py` (TRL-native dict-form multi-env) runners. Default base: `Nanthasit/sakthai-context-1.5b-merged`. |
| [`openenv-multi-catalog-training/`](openenv-multi-catalog-training/) | **Catalog run** — trains one small model across all 8 `openenv/*` catalog environments (echo, sudoku, coding, chat, atari, openspiel, repl, sumo) in a single GRPO run via the multi-environment pattern. |

Each directory has its own README with setup, run-elsewhere instructions, and
known rough edges.

## Status — Verified Contract & Improved (2026-08-03)

Both workspaces are updated with verified GRPO training contracts:

- **Local Verification Contract (`verify_grpo_contract.py`)**: Run `python3 verify_grpo_contract.py` locally to verify reward function logic, TRL `GRPOTrainer` compatibility, and `vllm_server_host`/`vllm_server_port` parameters on CPU before launching GPU runs.
- **Target Model (`Nanthasit/sakthai-context-7b-tools`)**: Based on empirical findings in `sakthai-agentic-eval-train/FINDINGS.md`, **7B** is the primary viable target for GRPO RL training (showing genuine rollout reward signal and `grad_norm` 0.25–0.49).
- **`openenv-custom-training`**: Updated to fix `vllm_server_host` and `vllm_server_port` parameter handling in both single-env (`train.py`) and multi-env (`multi_env.py`) runners.
- **`openenv-multi-catalog-training`**: Configured across all 8 `openenv/*` catalog environments (echo, sudoku, coding, chat, atari, openspiel, repl, sumo).

To launch a GPU training job on Hugging Face Jobs:
```bash
hf jobs uv run --detach --name grpo-7b-run --flavor a100-large --secrets HF_TOKEN --timeout 60m \
  -e TRAIN_MODE=lora16 -e TRAIN_BASE=Nanthasit/sakthai-context-7b-tools \
  -e TRAIN_MAX_STEPS=150 -e TRAIN_EPISODES=8 -e TRAIN_MAX_COMPLETION=1024 \
  -e TRAIN_PUSH_TO=Nanthasit/sakthai-context-7b-tools-grpo \
  grpo_train_pilot.py
```

## License

[Apache-2.0](LICENSE), consistent with the SakThai family's published artifacts
(see the `LICENSE` on the base models and the Kaggle-notebooks dataset).
