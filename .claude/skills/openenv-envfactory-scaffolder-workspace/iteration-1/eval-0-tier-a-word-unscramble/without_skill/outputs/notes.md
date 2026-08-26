# Word-Unscramble GRPO Environment — Notes

## What was added

1. **`openenv-custom-training/env_word_unscramble.py`** — new Tier A environment
   (inline plain Python, no server/Docker). Mirrors the structure of the
   reference `env_simple_task.py`.

2. **`openenv-custom-training/train.py`** — updated to dispatch the new env.
   - Added `word_unscramble` to `_select()` (imports from `env_word_unscramble`).
   - Added `word_unscramble` to the `--env` argparse `choices`.
   - Updated the module docstring's usage examples and the error message.

## Tier decision: Tier A

No tool method exec/eval/shells out on model-produced strings — `submit(word)`
just compares strings, `hint()` reveals a character from stored state. Safe
plain-Python. No sandboxed server needed.

## Contract compliance (per CLAUDE.md)

- `__init__(self)` takes no arguments.
- `reset(**kwargs)` receives `target` (scoring-only) and `scrambled`
  (also used in the first observation) as keyword args.
- Named tools: `submit(word: str)`, `hint()` — not a generic `step()`.
- Google-style docstrings with `Args:` blocks on tools that take arguments;
  `hint()` takes no args so its docstring omits the `Args:` block (an empty
  `Args:` block can trip up `_parse_google_format_docstring`).
- Episode state on `self`: `self.reward`, `self.done`, `self._attempts`,
  `self._target`, `self._scrambled`, `self._revealed`.
- Reward function signature: `reward_func(environments, **kwargs) -> list[float]`
  reading `env.reward` back off each instance. Binary (1.0 / 0.0), outcome-based.
- Cap: `MAX_ATTEMPTS = 6` (the user's requested cap). Only `submit()` counts
  against it; `hint()` does not (matches the intuition that "attempts" are
  guesses).
- Raising `ValueError` used for post-`done` calls — TRL catches and feeds
  message back to the model.
- Dataset `prompt` is a conversational list of `{"role", "content"}` dicts,
  not a bare string (a bare string would crash `prompt[-1]["content"]`).
- Deterministic dataset (fixed word bank + seeded `random.Random(0xC0FFEE)`)
  for reproducibility, mirroring `env_simple_task.py`'s `(i*7) % 100` trick.

## Default model

`DEFAULT_MODEL = "Nanthasit/sakthai-context-1.5b-tools"` — a SakThai *tools*
model (per the request). The 1.5B variant is a reasonable middle ground for
a task of this difficulty; `--model` / `TRAIN_BASE` overrides. Note the tools
models ship a tool-calling chat template (unlike `-merged` variants), which
GRPOTrainer requires when `environment_factory=` is set.

Per FINDINGS.md (quoted in CLAUDE.md), for the harder hermes agentic tasks
`0.5b-tools` gives zero rollout variance and `7b-tools` is the only viable
GRPO target. Word-unscramble is much easier than the hermes suite, so `1.5b`
should still sample enough successes for GRPO's group-relative advantage to
be non-zero.

## Smoke tests run (in this session)

Ran a small in-process exercise against the local `env_word_unscramble.py`:

- Dataset builder returns columns `prompt`, `target`, `scrambled`; `scrambled`
  is always a permutation of `target` and never equal to it.
- `reset()` returns an initial observation string that names the scrambled
  word, the letter count, the attempt cap, and the `hint()` availability.
- `submit()` returns "Not quite" feedback with match count, "Wrong length"
  feedback on length mismatch, and "Correct!" on the true target — setting
  `self.reward = 1.0`, `self.done = True`.
- `hint()` reveals leftmost unrevealed position on each call; two successive
  calls returned positions 0 then 1. Deterministic (leftmost-first) —
  reproducible rollouts.
- Post-`done` tool calls raise `ValueError("Episode already ended.")`.
- Attempt cap: 6 wrong same-length submits terminates the episode with
  reward 0.0.
- `reward_func([solved_env, failed_env])` returned `[1.0, 0.0]`.

I could not run `transformers.get_json_schema` on the tool methods
(`transformers` isn't installed in this CPU checkout, matching the CLAUDE.md
description of the repo). The docstring style is copied from
`env_simple_task.py` which is known to work.

## Files written

Under `outputs/openenv-custom-training/`:
- `env_word_unscramble.py` (new)
- `train.py` (modified copy of the real `openenv-custom-training/train.py`)

Nothing under the real `/home/user/openenv-rl-training/openenv-custom-training/`
was modified. No commits.
