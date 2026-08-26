# Scaffold notes — word_unscramble env

## Tier picked: **Tier A** (inline plain Python, no server)

Rationale: no tool method executes strings the model produced. `guess(word)`
compares the submitted string to the hidden target with `sorted()` equality
and `==`; `hint()` reads the target and writes to a set of revealed
positions. There is no shell command, no `exec`, no filesystem write
derived from model output, and no template that gets rendered into a path.
That is the exact condition Tier A is legitimate for — the container
isolation Tier B provides would buy nothing here.

## What the module ships

Single file: `openenv-custom-training/env_word_unscramble.py`.

- `WordUnscrambleEnv` — `__init__(self)` no args; `reset(**kwargs)` reads
  the scoring-only `target` column, computes a deterministic scramble
  seeded by the target word, returns the scramble as the first observation.
- Tools:
  - `guess(word: str)` — counts as an attempt only when length + letter
    multiset match the scramble; rejects malformed inputs with
    `ValueError` (feedback flows back to the model, does not burn the
    budget).
  - `hint()` — reveals the leftmost unrevealed letter; refuses when only
    one unknown remains so the model must always deduce at least one
    position.
- `MAX_ATTEMPTS = 6` guess budget per the user request. A non-converging
  rollout terminates cleanly with `self.done = True` and reward 0.0.
- `reward_func(environments, **kwargs)` — binary, outcome-based, reads
  `env.reward` back off each instance.
- `build_dataset(n_episodes=64)` — deterministic pick from a 20-word bank
  via `(i * 7) % len(WORDS)`. `prompt` is a list of `{role, content}`
  dicts as required (a bare string crashes on the first batch).
- Module-level exports: `ENVIRONMENT_FACTORY`, `REWARD_FUNCS`,
  `TRAIN_DATASET`, `DEFAULT_MODEL = "Nanthasit/sakthai-context-1.5b-tools"`
  (a tools model, not a -merged base — the latter has no tool-calling
  chat template and GRPOTrainer would raise).

## Wiring into `train.py`

Two edits to `openenv-custom-training/train.py`:

1. Extend the `_select()` dispatch (around line 97) with a new branch:

   ```python
   elif args.env == "word_unscramble":
       from env_word_unscramble import (DEFAULT_MODEL, ENVIRONMENT_FACTORY,
                                        REWARD_FUNCS, TRAIN_DATASET)
   ```

2. Extend the `--env` argparse `choices=` list (around line 124) and
   update the "unknown --env" error message (around line 114) to include
   `word_unscramble`:

   ```python
   choices=["simple", "agent_tools", "browsergym", "word_unscramble"],
   ```

That's it — no requirements.txt change (only `datasets` is needed, which
`train.py` already imports transitively). Run with:

```
python train.py --env word_unscramble --vllm-mode colocate
```

## Validator output

```
Validating .../env_word_unscramble.py …
  ok   exports ENVIRONMENT_FACTORY
  ok   exports REWARD_FUNCS
  ok   exports TRAIN_DATASET
  ok   DEFAULT_MODEL = 'Nanthasit/sakthai-context-1.5b-tools'
  ok   ENVIRONMENT_FACTORY class: WordUnscrambleEnv
  ok   WordUnscrambleEnv.__init__ takes no args
  ok   tool methods: ['guess', 'hint']
  ok   tool `guess` has docstring + Args + type hints
  ok   reward function `reward_func` signature ok
  ok   dataset has 4 rows; prompt is well-formed

OK: 10 check(s) passed, 0 warning(s).
```

(`hint()` has no parameters, so the validator correctly does not flag its
docstring for an `Args:` block — the block is only mandatory when
parameters exist.)
