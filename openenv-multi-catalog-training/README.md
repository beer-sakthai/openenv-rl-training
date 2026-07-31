# Multi-catalog OpenEnv + TRL GRPO training

Trains one small (~0.6B) model across all 8 `openenv/*` catalog
environments in a single GRPO run, using TRL's `environment_factory` and
the multi-environment (meta-environment) pattern.

## Read this first

This was scoped explicitly to include `atari_env` (pixel-based) and
`sumo_rl_env`/`openspiel_env` (numeric-vector observations) alongside the
text/tool-calling environments, on a 0.6B **text-only** model. That's a
real modality mismatch, not a style choice — a text model cannot see
pixels. `train_multi_env.py`'s module docstring spells out the specific
workarounds (RAM-byte text proxy for Atari, raw-token bridging for
`chat_env`) and their limits. Expect the atari/chat/openspiel/sumo reward
curves to lag the text-native tasks (echo/sudoku/coding/repl) — that's the
mismatch showing up in the metrics, not necessarily a bug.

If you'd rather validate the pipeline first, comment out everything except
`echo_reward/"echo"` in `REWARD_FUNCS`/`build_dataset` and confirm the
end-to-end loop (server → tool call → reward → GRPO step) works before
re-enabling all 8.

## Setup

This machine has no GPU / working Docker daemon / torch install — none of
this has been run. On your actual training machine (1 GPU, per your
earlier answer):

```bash
pip install -r requirements.txt
bash run_servers.sh          # starts all 8 env containers, ports 8001-8008
docker ps                    # verify all 8 are Up
python train_multi_env.py --vllm-mode colocate
```

If any `docker run` in `run_servers.sh` fails because the image tag has
moved, open that Space's page → "⋮" → "Run locally" to get the current
`registry.hf.space/...` tag and swap it in.

## What to watch

TRL logs `train/reward_func_0` .. `train/reward_func_7`, one per
environment (order matches `REWARD_FUNCS` in `train_multi_env.py`: echo,
sudoku, coding, chat, atari, openspiel, repl, sumo). Watch those
individually — the combined `train/reward` alternates across all 8 tasks
batch-to-batch and reads as noisy even when training is healthy.

## Known rough edges to fix before this is production-quality

- `coding_env`'s task is a placeholder (print `17 * 23`) — swap
  `CODING_TASK_PROMPT`/`CODING_EXPECTED` in `train_multi_env.py` for the
  real coding task and a real correctness check.
- `chat_env`'s action schema (`ChatAction.tokens`) was inferred from the
  Space README summary, not verified against running code — the
  tokenizer-bridging in `chat_reply()` is the first thing to check if that
  environment errors out.
- `atari_press` fixes the game via `ATARI_GAME` (`pong` by default) at
  reset — multi-game training would need the game name in the dataset's
  routing instead.
- No image tags were verified live (this container can't reach Docker) —
  confirm each via the Space's "Run locally" panel before trusting
  `run_servers.sh` as-is.
