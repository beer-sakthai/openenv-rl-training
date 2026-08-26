---
name: openenv-envfactory-scaffolder
description: Scaffolds a compliant TRL `environment_factory` class plus its matching reward function and dataset builder from a natural-language task description, for the openenv-rl-training repo. Picks Tier A (inline plain-Python env, no server) vs Tier B (sandboxed OpenEnv server) automatically based on whether any tool would execute model-produced strings. Use whenever the user wants to add a new OpenEnv environment, an OpenEnv custom env, a GRPO task, a TRL tool-calling environment, or asks anything like "add an environment for X", "scaffold a new env", "make a training task for Y", "wire up a new GRPO environment", "create an env_factory for …", or "port this task idea into openenv-custom-training/", even if they don't say the word "scaffold" — this is the file that turns a task idea into a train.py-ready module without the contract landmines.
---

# OpenEnv env-factory scaffolder

## What this does

The `openenv-rl-training` repo trains models with TRL's
`GRPOTrainer(environment_factory=...)`. That contract is small on the surface
but has a dozen sharp edges (docstring style, tool-naming, episode caps,
routing columns, security tiering, GRPOConfig fields). Getting any of them
wrong tanks a run silently — a missing `Args:` block raises at trainer
construction, a wrong dataset column type crashes on the first batch, and a
generic `step(action)` tool starves GRPO of signal.

This skill takes a natural-language task description and produces a **complete
module** that plugs straight into `train.py` — the environment class, the
reward function, and the dataset builder — following the exact conventions
`env_simple_task.py` and `agent_tools/wrapper.py` already use in this repo.

## When to trigger

Trigger this skill whenever the user wants to add or scaffold anything that
plugs into `GRPOTrainer(environment_factory=...)`:

- "Add an env for [task]" / "scaffold an OpenEnv env for [task]"
- "Make a new GRPO training task"
- "Port this idea into openenv-custom-training/"
- "Write an environment_factory for [X]"
- "Add a new environment to multi_env.py"
- "Set up a sandboxed env that runs shell / Python / SQL / [anything the
  model produces]" → Tier B

Do **not** trigger for edits to an existing env's business logic (that's
plain editing), or for HF Jobs / eval scripts.

## The two tiers — pick before writing anything

The Tier A vs Tier B choice is a **security decision**, not a stylistic one.
Read the user's task description and ask: **does any tool method execute
strings the model produced?** Shell commands, Python code, SQL queries,
regex patterns fed to `re.sub`, template strings rendered into a filesystem
path — all yes.

- **No** → Tier A. Inline plain Python in the training process. Cheap,
  legitimate. Reference: `references/tier-a-reference.py`.
- **Yes** → Tier B. Sandboxed OpenEnv server in a container. GRPO
  exploration *will* emit off-distribution actions and you don't want
  those running in your training process. Reference:
  `references/tier-b-reference.py`.

There is deliberately no "run in-process with a string blocklist" tier. A
blocklist is a false sense of security; the policy will find a spelling you
didn't block. Tier A is safe *because the tools' Python bodies cannot execute
model output*, not because you tried to filter what the model wrote.

If in doubt, ask the user with `AskUserQuestion` — it's a one-way door.

## Workflow

1. **Read the contract.** Open `references/contract.md` before writing a
   line — it has every rule TRL enforces and every trap the codebase has
   already hit. Don't try to re-derive them from the reference files alone.

2. **Pick the tier** using the rule above. If Tier B, also decide on the
   task registry — one class handling multiple named acceptance predicates
   keyed by a `task` column, like `sandbox_env.py`'s TASKS dict, is almost
   always the right shape.

3. **Copy-adapt the reference file** for the chosen tier:
   - Tier A → `references/tier-a-reference.py` → `openenv-custom-training/env_<name>.py`
   - Tier B → `references/tier-b-reference.py` → `openenv-custom-training/<name>_tools/`
     (server-side + wrapper split). For Tier B, mirror
     `agent_tools/`'s three-piece layout: `wrapper.py` (TRL-facing),
     `server/sandbox_env.py` (task logic), `models.py` (Action/Observation
     dataclasses), `client.py` (HTTP shim).

4. **Fill in the task-specific bits**, keeping every convention listed in
   `references/contract.md` intact:
   - `__init__(self)` — no args, initialize `self.reward = 0.0` and
     `self.done = False`
   - `reset(**kwargs)` — extract dataset columns, reset episode state,
     return the first observation (a string) or `None`
   - **Named tools** with Google-style docstrings (see contract §Tools).
     `guess(number: int)`, not `step(action)`. `Args:` block is mandatory.
   - **Episode cap** (`MAX_ATTEMPTS`, `STEP_LIMIT`) — a non-converging
     rollout must terminate with a clean 0.0
   - **Binary reward**, outcome-based, judged on final state (`env.reward`)
   - **Dataset**: `prompt` is a **list of `{role, content}` dicts**, not a
     plain string; scoring-only columns (`target`, `task`) go alongside
   - **Deterministic** dataset construction (`target = (i * 7) % 100`),
     never `random.randint`, so runs reproduce
   - Export `ENVIRONMENT_FACTORY`, `REWARD_FUNCS`, `TRAIN_DATASET`,
     `DEFAULT_MODEL` at module level so `train.py`'s dispatch table sees
     the module

5. **For Tier B only**, additionally:
   - `SUPPORTS_CONCURRENT_SESSIONS = True` as a **class attribute** on the
     `Environment` subclass — module-level does not work; the server calls
     `getattr(env_cls, ...)` on it
   - `create_app(..., max_concurrent_envs=N)` with `N >= num_generations`
   - Per-step guards on any exec path: timeout, output caps, cwd confined
     to a per-episode scratch dir with `realpath` escape checks, env
     stripped to `PATH`/`HOME`/`LANG`. No command blocklist. See
     `sandbox_env.py`'s `_confine` / `_run_command` for the pattern.
   - Container flags recommended in `references/contract.md` (§Tier B
     runtime): `USER nobody`, `docker run --network none --pids-limit 128`

6. **Validate** with `python scripts/validate_env.py <path/to/env.py>`.
   This is a CPU-only check that catches the frequent failure modes before
   `train.py` does: missing `Args:` block (raises `DocstringParsingException`
   at trainer construction), non-conversational `prompt`, `__init__` taking
   args, `SUPPORTS_CONCURRENT_SESSIONS` at module level instead of on the
   class, missing module-level exports. Run it before showing the file to
   the user.

7. **Optionally smoke-test** with `python scripts/smoke_test.py
   <path/to/env.py>`. Drives one episode locally — calls `reset`, then
   invokes each named tool with a plausible argument, then reads back
   `env.reward` / `env.done`. Catches runtime crashes the static validator
   can't see. Requires `datasets`; skips gracefully otherwise.

8. **Show the user** the generated files with a one-paragraph note that
   states the tier you picked and why, plus the exact `train.py` invocation
   they can use to run it (e.g., `python train.py --env <name>`, or
   `--env-registry <path>` if they want it out-of-tree).

## Why every rule matters

Don't leave "must have `Args:` block" as a rule the model just memorises —
explain that TRL calls `transformers.get_json_schema` on each public method
and it raises `DocstringParsingException` when any parameter lacks an
`Args:` entry. Same for `SUPPORTS_CONCURRENT_SESSIONS`: the server does
`getattr(env_cls, "SUPPORTS_CONCURRENT_SESSIONS", False)` and refuses
`max_concurrent_envs > 1` without it. When the model understands the
mechanism, it stops accidentally regressing.

`references/contract.md` is written this way throughout — every rule has
its "why" alongside so the generated code isn't a cargo-cult of the
reference files.

## Repo integration

Generated files land under `openenv-custom-training/` and are wired in
through `train.py`'s existing dispatch pattern (`--env <name>` maps to the
module). For multi-environment runs, they also plug into `multi_env.py`
(dict-form factory, routing by an `environment` column) or
`train_multi_env.py` (meta-class routing by a `env` column). The routing
column name is not consistent across the repo — pick one that matches the
file the env is being added to and stay consistent within that file. See
`references/contract.md` §Multi-env routing for the mapping.

## What this skill deliberately does not do

- **It does not run training.** GRPO needs a GPU box and the heavy stack
  (`torch`, `trl`, `vllm`, ~GB); that runs on HF Jobs, not here. This skill
  produces the module and stops.
- **It does not add container images** for Tier B. It scaffolds the
  server-side Python; building the OpenEnv image (`openenv init` +
  `openenv build`) and pushing to `registry.hf.space` is a separate step.
- **It does not modify `CLAUDE.md`, `README.md`, or existing envs.** If
  the new env belongs in `multi_env.py`'s dict or the meta-class registry,
  the scaffold notes what to add — the user or a follow-up edit does it.
