# CLAUDE.md

Guidance for AI assistants working in this repository.

## What this repo is

Four **independent workspaces** for GRPO (reinforcement-learning) fine-tuning of the
SakThai model family with [Hugging Face OpenEnv](https://github.com/huggingface/OpenEnv)
and [TRL](https://huggingface.co/docs/trl)'s `GRPOTrainer(environment_factory=...)`
multi-turn tool-calling loop.

It is **not an installable package**: no `setup.py`/`pyproject.toml`, no CI (`.github/`
does not exist), no linter or formatter config, no test suite beyond the two root-level
contract checks. Each subdirectory has its own `requirements.txt` and is run directly
with `python <script>.py`, `uv run`, or `hf jobs uv run`.

The heavy work (GRPO training, evaluation) runs **elsewhere** — HF Jobs, Colab/Kaggle,
or a rented GPU box. This checkout has no GPU and typically no `torch`/`trl`/`datasets`
installed. Several READMEs say "written, not run" or "design-only"; those statements are
accurate and must stay accurate (see *Documentation conventions*).

## Layout

| Path | What it is |
|---|---|
| `openenv-custom-training/` | **Custom environments.** Tier A (`env_simple_task.py`, inline plain Python), Tier B (`agent_tools/`, sandboxed OpenEnv server in Docker), and BrowserGym MiniWoB++ (`train.py --env browsergym`). Runners: `train.py` (single env), `multi_env.py` (TRL-native dict-form multi-env). |
| `openenv-multi-catalog-training/` | **Catalog run.** One ~0.6B model across all 8 `openenv/*` catalog envs (echo, sudoku, coding, chat, atari, openspiel, repl, sumo) in one GRPO run via a meta-environment class. `a2a_agent/` exposes the same 8 envs as [A2A protocol](https://a2a-protocol.org/) skills, independent of training. |
| `sakthai-agentic-eval-train/` | **The eval + train pipeline that actually ran.** As-run HF Jobs scripts (bench eval, agentic eval, SFT bootstrap, GRPO pilot), a self-contained Colab/Kaggle notebook, and `FINDINGS.md` — the durable empirical record. |
| `browsergym-space/` | Dockerfile + Space card for the BrowserGym OpenEnv server deployed at [`Nanthasit/browsergym-env`](https://huggingface.co/spaces/Nanthasit/browsergym-env) (`https://nanthasit-browsergym-env.hf.space`). |
| `verify_grpo_contract.py`, `test_browsergym_contract.py` | CPU-only contract checks — the only things runnable in this checkout. |
| `PLAN.md` | The improvement plan the recent commits implement. Checkboxes are stale (unticked despite the work landing); treat the file as intent, not status. |

`.gitattributes` at the root is Hugging Face's auto-generated LFS config, merged in from
a Space repo. Leave it alone.

## Running things locally (what actually works here)

```bash
python3 verify_grpo_contract.py          # passes; skips the TRL section when trl is absent
uv run --with datasets --with pytest pytest test_browsergym_contract.py   # 3 passed
```

`test_browsergym_contract.py` mocks `trl` (`sys.modules['trl'] = MagicMock()`) but **not**
`datasets` — plain `pytest test_browsergym_contract.py` fails at import with
`ModuleNotFoundError: No module named 'datasets'` unless `datasets` is installed. It also
does `sys.path.append("openenv-custom-training")`, so it only works from the repo root.

Run both before touching anything under `openenv-custom-training/`. Everything else needs
a GPU box; do not attempt to run training here.

## The TRL `environment_factory` contract

This is the core convention every environment class in the repo follows. Getting it wrong
is the most common failure mode.

- **`__init__(self)` takes no arguments.** TRL instantiates one environment per generation
  slot: `EnvironmentFactory()`, never `EnvironmentFactory(cfg)`. Configuration comes from
  module-level constants or env vars.
- **`reset(**kwargs)` receives every dataset column as a keyword argument** (except the
  routing control column in dict-factory form). Return a string to give the model its first
  observation, or `None`.
- **Every public method other than `reset` becomes a callable tool**, named after the
  method. The schema is generated from type hints + docstring by
  `transformers.get_json_schema`, which raises `DocstringParsingException` if any parameter
  lacks an `Args:` entry. **Google-style docstrings with a full `Args:` block are mandatory;
  one-line docstrings fail.** The docstring is the interface the model reads — write it as a
  tool description, not as a code comment.
- **Prefer named tools** (`guess(number: int)`, `run_command(command: str)`) over a single
  generic `step(action)`.
- **Episode state lives on `self`** (`self.reward`, `self.done`); reward functions read it
  back off the instance after the episode.
- **Raising an exception rejects a call** — TRL catches it and feeds the message back to the
  model as the tool result. Used for out-of-range arguments and for out-of-turn tool calls
  in multi-environment classes.
- **Cap the episode.** Every environment has a step/attempt limit (`MAX_ATTEMPTS = 10`,
  `STEP_LIMIT = 15`) so a non-converging rollout terminates with a clean `0.0` instead of
  looping.

### Reward functions

Signature is `reward_func(environments, **kwargs) -> list[float]`, forwarding `env.reward`.
Rewards are **binary (1.0/0.0) and outcome-based** — judged on final state, not the path
taken. GRPO ranks within a group, so only the ordering a reward induces matters; outcome-only
rewards let the model find strategies you didn't script. In Tier B the *server* owns the
state and therefore owns the verdict; the wrapper forwards `reward`/`done` untouched.

In multi-environment runs, one reward function **per environment**, each returning `None`
for episodes that belonged to another (TRL turns `None` into `NaN` and aggregates each
reward over its own episodes only).

### Dataset conventions

- **`prompt` must be conversational** — a list of `{"role", "content"}` dicts, not a plain
  string. TRL's tool-calling GRPO does `prompt[-1]["content"]`; a bare string fails with
  `TypeError: string indices must be integers`.
- Scoring-only columns (`target`, `task`) are passed to `reset(**kwargs)` and never shown to
  the model.
- Routing column names are **not consistent across workspaces** — check before copying:
  `multi_env.py` uses `environment` (TRL's convention for dict factories, a control field
  not forwarded to `reset`), `train_multi_env.py`'s meta-class uses `env`, `agent_tools`
  uses `task`. Keep each file self-consistent rather than unifying them casually.
- Datasets are built deterministically (`target = (i * 7) % 100`), not randomly, to keep
  runs reproducible.

### GRPOConfig gotchas

- **vLLM server mode fields are `vllm_server_host` + `vllm_server_port`.** `vllm_server_url`
  does not exist and crashes at construction. Fixed in `train.py` and `multi_env.py`;
  **`openenv-multi-catalog-training/train_multi_env.py:main()` still passes
  `vllm_server_url`** — a real outstanding bug, PLAN.md Task 2 only covered the custom
  workspace.
- **`max_completion_length` caps tokens across the WHOLE multi-turn episode** (every
  generation plus every tool result, summed) — not one turn. Episodes truncating mid-task is
  the first thing to suspect, and a too-small cap guarantees reward 0, which starves GRPO of
  signal entirely.
- 1 GPU → `--vllm-mode colocate`. 2+ GPUs → `trl vllm-serve` on one, then `--vllm-mode server`.
- Hard requirements when `environment_factory` is used: **`transformers>=5.2.0`** (GRPOTrainer
  raises below it), **`jmespath`** (undocumented dep, needed for tool-response parsing), and a
  base model whose **chat template supports tool calling** (GRPOTrainer validates and raises
  otherwise — this is why `sakthai-context-0.5b-merged`, which ships no chat template, is not
  a valid base).
- Concurrency for server-backed envs: GRPO opens one session per generation, so the server
  needs `SUPPORTS_CONCURRENT_SESSIONS = True` as a **class attribute** on the `Environment`
  subclass, and `create_app(..., max_concurrent_envs=N)` with `N >= num_generations`.

## Model choice — settled empirically, don't re-litigate

`sakthai-agentic-eval-train/FINDINGS.md` is the record. The headline:

> **GRPO can only reinforce successes the model samples during rollouts.** `0.5b-tools`
> solves the hermes tasks ~0% of the time → every rollout in a group fails → zero reward
> variance → zero advantage → zero gradient. Confirmed at 40 steps:
> `frac_reward_zero_std: 1`, `grad_norm: 0`, weights byte-identical. An SFT bootstrap lifted
> eval to 1/6 but rollout variance stayed zero, and cost ~13 points of single-shot accuracy.
> **`Nanthasit/sakthai-context-7b-tools` (3/6 agentic, reward ~0.05, `grad_norm` 0.25–0.49)
> is the only viable GRPO target in this family.**

Defaults in the code reflect that: `train.py --env browsergym` and `multi_env.py` default to
`sakthai-context-7b-tools`; the Tier A/B modules still declare
`DEFAULT_MODEL = "Nanthasit/sakthai-context-1.5b-merged"` (their tasks are much easier).
`--model` / `TRAIN_BASE` overrides everywhere.

Other findings worth honoring when editing eval or training code:

- **Eval a trained model in the exact prompt format it was trained on.** TRL trains with the
  model's native `apply_chat_template(tools=...)`; the bench harness uses a hand-rolled
  ChatML renderer. Mismatching them made an SFT checkpoint look like 0/6 when it was 1/6.
  `eval_hermes_env.py` exposes `SAK_RENDER=native|handrolled` for exactly this — use `native`
  for SFT/GRPO-trained checkpoints. A suspicious `0/N` is a cue to read a raw transcript, not
  to conclude.
- **Cast merged models to bf16 before saving.** The as-run `grpo_train_pilot.py` saves fp32
  (~2x size, one 30.5GB checkpoint); the fix is applied in `sakthai_grpo_colab.ipynb`, which
  is the preferred starting point for a fresh run.
- GRPO from a bare LoRA adapter needs **merge-in** (adapter → base → local full-model dir so
  vLLM can load it) and **merge-out** (GRPO LoRA → standalone bf16 model for a usable push).
- The `hermes-tool-use-rl-env` environment is driven **in-process** (plain subprocess/tempdir
  Python) in every script here; its shipped Docker/`client.py`/WebSocket path is broken
  against `openenv==0.4.1` and is bypassed deliberately. Don't "fix" the scripts by routing
  them back through it.

## Security model

The one decision that matters when adding an environment: **does a tool method execute
strings the model produced?**

- **No →** Tier A shape (`env_simple_task.py`): inline plain Python in the training process.
  Cheap and legitimate.
- **Yes →** Tier B shape (`agent_tools/`): a real OpenEnv server in a container. GRPO
  exploration *will* emit off-distribution actions; those must not run in your training
  process.

Isolation is the **container**, not string blocklists — a blocklist gives false confidence
and the policy will find a spelling you didn't block. There is deliberately no command
blocklist in `server/sandbox_env.py`. On top of the container it adds per-step guards:
10s command timeout, stdout/stderr caps, cwd confined to a per-episode scratch dir with
`realpath` escape checks, and an env stripped to `PATH`/`HOME`/`LANG`. Recommended container
flags: `USER nobody`, `docker run --network none --pids-limit 128`.

Two places carry explicit "don't deploy this as-is" warnings that must be preserved:
`sakthai-agentic-eval-train/scripts/eval_hermes_env.py` runs model-generated shell commands
in-process (ephemeral job runners only, trusted models only), and
`openenv-multi-catalog-training/a2a_agent/` exposes 8 environments — one of which executes
arbitrary code — over plain HTTP with no auth or rate limiting.

## Workflows

### HF Jobs (paid; needs a payment method — jobs returned `402 Payment Required` at time of writing)

Scripts carry **PEP 723 inline dependency headers** (`# /// script ... # ///`) so
`hf jobs uv run` resolves deps without a requirements file. Keep those headers in sync when
adding an import.

```bash
# GRPO on BrowserGym MiniWoB++
hf jobs uv run --detach --flavor a100-large --secrets HF_TOKEN --timeout 1h \
  -e TRAIN_MODE=lora16 -e TRAIN_BASE=Nanthasit/sakthai-context-7b-tools \
  -e TRAIN_MAX_STEPS=150 -e TRAIN_EPISODES=8 -e TRAIN_MAX_COMPLETION=1024 \
  -e TRAIN_PUSH_TO=Nanthasit/sakthai-context-7b-tools-grpo \
  train.py --env browsergym --browsergym-task click-button

# agentic eval
hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \
  --env SAK_MODELS=Nanthasit/sakthai-context-7b-tools \
  sakthai-agentic-eval-train/scripts/eval_hermes_env.py
```

Sizing: eval and 0.5B training fit `l4x1`; 7B bench eval and 7B GRPO need `a100-large`
(80GB) — 7B bench OOMs on `l4x1` at batch 16.

**Every CLI flag in `train.py` also reads a `TRAIN_*` environment variable** as its argparse
default (`TRAIN_ENV`, `TRAIN_BASE`, `TRAIN_VLLM_MODE`, `TRAIN_MAX_COMPLETION`, `TRAIN_EPISODES`,
`TRAIN_PUSH_TO`, …). That exists so the jobs dispatcher can drive the script without
translating env vars into CLI args. Preserve the pattern when adding a flag.

### Free GPU

`sakthai-agentic-eval-train/sakthai_grpo_colab.ipynb` is the consolidated, self-contained
pipeline (install → auth → config → in-process env → merge-in → GRPO with `use_vllm=False`
for a T4 → bf16 merge-out → 6-task eval). Prefer it over the as-run scripts for a real run.
It is linked by Colab/Kaggle badges that point at `main` on GitHub — moving or renaming the
notebook breaks those badges in both READMEs.

### Local Docker servers

```bash
# Tier B sandbox (custom workspace)
openenv init agent_tools && openenv build
docker run -d -p 8001:8000 --platform linux/amd64 registry.hf.space/<you>/agent_tools:latest
AGENT_TOOLS_URL=http://localhost:8001 python train.py --env agent_tools --vllm-mode colocate

# all 8 catalog envs, ports 8001-8008
bash openenv-multi-catalog-training/run_servers.sh
```

Port 8001+ is used deliberately so 8000 stays free for a colocated vLLM server. If a
`docker run` fails on a moved image tag, get the current `registry.hf.space/...` tag from
the Space page's "⋮ → Run locally" panel.

### Reading training metrics

TRL logs `train/reward_func_0..N`, one per reward function, in `REWARD_FUNCS` order. **Watch
those individually** — the combined `train/reward` alternates between tasks batch to batch
and reads as noisy even when training is healthy.

Before spending GPU time on a new environment, drive a few episodes by hand and confirm a
capable model scores above random. Both workspaces' environments are callable directly.

## Documentation conventions

The prose in this repo is unusually careful, and that is deliberate. Match it:

- **Verification claims carry a date and a method** — "verified end-to-end against the real
  `openenv==0.4.1` on 2026-07-31", "trl 1.9.2 source read". Never upgrade an unverified
  claim to a verified one, and never delete a "this was not run" caveat, unless you actually
  ran it in this session.
- **Known-gaps sections are load-bearing.** `coding_env`'s placeholder task, `chat_env`'s
  inferred action schema, unverified Docker image tags, the untested `a2a-sdk` method names —
  each is flagged where it lives. If you fix one, remove its caveat; if you touch nearby
  code, leave it.
- Module docstrings do the explaining. `train_multi_env.py`, `agent_tools/README.md`, and
  `env_simple_task.py` are the reference examples: they state the contract, the caveats, and
  the reasoning, not just the API.
- `FINDINGS.md` is the durable empirical record. Append to it; don't quietly restate its
  numbers elsewhere in a way that could drift.

## Git conventions

- Default branch is `main`; the remote is `beer-sakthai/openenv-rl-training`.
- Work happens on `claude/<topic>-<suffix>` branches, merged to `main` via PR.
- Commit subjects are Conventional-Commits-flavored with an optional scope:
  `feat(grpo): …`, `fix(train): …`, `refactor(openenv): …`, `docs: …`.
- Secrets never land in the repo — `.gitignore` covers `.env`, `auth.json`,
  `.git-credentials`, `*.log`, `.eval_results/`. `HF_TOKEN` arrives via `--secrets HF_TOKEN`
  (HF Jobs), a Kaggle Secret, or an interactive paste in Colab.

## Known open items

- `train_multi_env.py` still passes the nonexistent `vllm_server_url` GRPOConfig field
  (PLAN.md Task 2, applied only to the custom workspace).
- `train.py --env browsergym` imports `browsergym_env`, which is in neither the PEP 723
  header nor `openenv-custom-training/requirements.txt`; the factory raises a hinted
  `ImportError` pointing at `pip install git+https://github.com/huggingface/OpenEnv.git`.
- `coding_env`'s task in both `train_multi_env.py` and `a2a_agent/` is a placeholder
  (`print(17 * 23)`) with a substring check for correctness.
- No catalog Docker image tag in `run_servers.sh` has been verified live.
- `a2a_agent/` has never been executed; the `TaskUpdater` method names were written from the
  published SDK pattern, not a live install.
- The 7B GRPO proof-of-signal run was 40 steps — long enough to show a gradient exists, not
  long enough to improve the model. A real run needs hundreds of steps.
