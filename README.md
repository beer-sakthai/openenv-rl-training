# OpenEnv + TRL GRPO training — SakThai family

Reinforcement-learning (GRPO) fine-tuning of SakThai-family models with
[Hugging Face OpenEnv](https://github.com/huggingface/OpenEnv) (sandboxed,
Gymnasium-style environments) and [TRL](https://huggingface.co/docs/trl)'s
`GRPOTrainer` `environment_factory` (multi-turn tool-calling RL loop).

Two companion workspaces:

| Directory | What it is |
|-----------|------------|
| `openenv-custom-training/` | **Custom environments** — Tier A (inline plain-Python logic, no server) and Tier B (sandboxed OpenEnv server for agent tool-use tasks), plus `train.py` (single-env) and `multi_env.py` (TRL-native dict-form multi-env) runners. Default base: `Nanthasit/sakthai-context-1.5b-merged`. |
| `openenv-multi-catalog-training/` | **Catalog run** — trains one small model across all 8 `openenv/*` catalog environments (echo, sudoku, coding, chat, atari, openspiel, repl, sumo) in a single GRPO run via the multi-environment pattern. |

Each directory has its own README with setup, run-elsewhere instructions, and
known rough edges.

## Status — validated design-only (2026-07-31)

Both workspaces are **design artifacts**: they were built and validated on a
box with no GPU, and have not been trained on. What was actually verified:

- **`openenv-custom-training`** — the Tier B server+client were run **live
  end-to-end** against the real `openenv==0.4.1` (uvicorn server, real
  WebSocket sessions: all three sandbox tasks complete with binary reward,
  path-escape refused, concurrent sessions isolated). The `trl==1.9.2`
  `environment_factory`/`GRPOConfig` contract was confirmed by reading the
  installed source (hard requirements: `transformers>=5.2.0`, `jmespath`,
  tool-calling chat template). Tier A's game was smoke-tested (bisection
  solves 15/15, random policy ~0.07).
- **`openenv-multi-catalog-training`** — written but **never run**; it needs
  8 environment containers (`run_servers.sh`) and a GPU. Read its README's
  caveats (Atari is a RAM-text proxy on a text-only model, chat_env's action
  schema was inferred, etc.) before trusting its results.

To train, copy a workspace to a GPU box (Kaggle / HF Jobs / rented), follow
its README, and run.

## License

Apache-2.0, consistent with the SakThai family's published artifacts
(see the `LICENSE` on the base models and the Kaggle-notebooks dataset).
