# CLAUDE.md

Guidance for AI assistants working in this repository.

## What this repo is

Five **independent workspaces** covering both halves of the SakThai training pipeline: the
supervised half (QLoRA on Qwen2.5 for tool-calling — `sakthai-sft-training/`) and the
reinforcement-learning half (GRPO with [Hugging Face OpenEnv](https://github.com/huggingface/OpenEnv)
and [TRL](https://huggingface.co/docs/trl)'s `GRPOTrainer(environment_factory=...)` multi-turn
tool-calling loop — the other four workspaces).

Consolidated on 2026-08-21 from two GitHub repos into one; the retired source was
`beer-sakthai/SakThai-Training` (deleted on GitHub 2026-08-21; full git history preserved
in the local archive at `/home/beern/archive/SakThai-Training`).

It is **not an installable package**: no `setup.py`/`pyproject.toml`. Each subdirectory has
its own dependencies (via `requirements.txt` or a PEP 723 inline header) and is run directly
with `python <script>.py`, `uv run`, or `hf jobs uv run`. The root `requirements.txt` is
**not** an install manifest for the pipeline — it is only the CPU dev/test toolchain
`verify-contracts.yml` installs (datasets, httpx, pydantic, pytest, ruff); its own header
explains why unifying the workspaces' pinsets there does not resolve.

There **is** a linter config (`.ruff.toml`) as of 2026-09-16, but a deliberately narrow one:
E9, F821 and F811 only — syntax errors, undefined names, redefinitions. Style rules stay off
because ~115 pre-existing F541/F401/F841 hits sit mostly in files that must not be touched
(the `cycle-100-v*.py` snapshots). Do not widen the select list without doing the cleanup
first, and never by reformatting a pinned snapshot.

CI (`.github/workflows/`): `verify-contracts.yml` runs entirely inside GitHub Actions — ruff,
`verify_grpo_contract.py`, and 38 pytest tests across the root contract file and both
workspace suites. Four workflows dispatch `hf jobs uv run` to Hugging Face and need
`HF_TOKEN` as a repo secret (`train`, `eval`, `lighteval`, `mcp-bench`); `monitor.yml`
does not dispatch anything — it reads the Hub API and writes a job summary.

Do not re-add the GitHub-suggested `pylint.yml` / `python-app.yml` / `python-package.yml` /
`super-linter.yml` / `cache.yml` / `label.yml` / `manual.yml` templates — the first five
hardcode `pip install -r requirements.txt` at the root and then run the whole tree through a
style linter, `manual.yml` was a "Hello World" echo (deleted 2026-09-16), and `label.yml`
(`actions/labeler@v4`) fetches its config from the base branch on `pull_request_target`, so
`.github/labeler.yml` on a feature branch is invisible until merged — the labeler check
fails on every first-time PR by construction, not by bug.

The heavy work (GRPO training, evaluation) runs **elsewhere** — HF Jobs, Colab/Kaggle,
or a rented GPU box. This checkout has no GPU and typically no `torch`/`trl`/`datasets`
installed. Several READMEs say "written, not run" or "design-only"; those statements are
accurate and must stay accurate (see *Documentation conventions*).

## Related repositories

Two sibling repos under `beer-sakthai`. Know which one owns what before you go looking:

- **`beer-sakthai/Sak-Family-Agent`** — the agent runtime (the `sakthai` package, six
  personas, memory store, MCP server, web API) plus its own `training/` HF Jobs
  definitions. Its `training/sakthai-7b-lora/train.py` pushes
  `Nanthasit/sakthai-context-7b-tools`, which is the GRPO base here. The two repos share
  **no code** and pin incompatible dependency sets — do not cross-import, and do not
  restate this repo's benchmark numbers over there (`FINDINGS.md` and the workspace
  READMEs are the durable records; see *Documentation conventions*).
- **`beer-sakthai/codeql-action`** — a fork of `github/codeql-action` carrying local
  dependency-advisory remediation against the action's own dev-dependency tree.
  `.github/workflows/codeql.yml` here pins **upstream**, not the fork.

## Layout

| Path | What it is |
|---|---|
| `sakthai-sft-training/` | **SFT half.** QLoRA training scripts (0.5B / 1.5B / 7B; `train-sakthai-1.5b-v2.py`, `train_qwen.py`, `train-sakthai-cpu.py`), 10-cycle data-augmentation loop (`cycle-100-v2..v10.py`, `augment-*.py`, `audit-and-fix-safety-quality.py`), cross-model + MCP-Bench + lighteval evaluators (`eval-*.py`), `sakthai-cycle-bench/` (155-row BFCL harness), ops tooling (`scripts/ops/`, `scripts/eval/`), self-contained SFT Colab notebook (`sakthai-1.5b-colab.ipynb`), publishing helpers (`push-*.py`, `create-*.py`). Was `beer-sakthai/SakThai-Training` before 2026-08-21. |
| `openenv-custom-training/` | **RL — custom environments.** Tier A (`env_simple_task.py`, inline plain Python), Tier B (`agent_tools/`, sandboxed OpenEnv server in Docker), and BrowserGym MiniWoB++ (`train.py --env browsergym`). Runners: `train.py` (single env), `multi_env.py` (TRL-native dict-form multi-env). |
| `openenv-multi-catalog-training/` | **RL — catalog run.** One ~0.6B model across all 8 `openenv/*` catalog envs (echo, sudoku, coding, chat, atari, openspiel, repl, sumo) in one GRPO run via a meta-environment class. `a2a_agent/` exposes the same 8 envs as [A2A protocol](https://a2a-protocol.org/) skills, independent of training. |
| `sakthai-agentic-eval-train/` | **The RL eval + train pipeline that actually ran.** As-run HF Jobs scripts (bench eval, agentic eval, SFT bootstrap, GRPO pilot), a self-contained Colab/Kaggle notebook, and `FINDINGS.md` — the durable empirical record. |
| `browsergym-space/` | Dockerfile + Space card for a BrowserGym OpenEnv server. **Not deployed** — `Nanthasit/browsergym-env` no longer exists (404, verified 2026-09-16). `train.py --env browsergym` defaults to upstream [`openenv/browsergym_env`](https://huggingface.co/spaces/openenv/browsergym_env); redeploy from here and pass `--browsergym-url` to use your own. |
| `.github/workflows/` | `verify-contracts.yml` (ruff + 38 CPU tests, runs in GH Actions); `train.yml`/`eval.yml`/`lighteval.yml`/`mcp-bench.yml` (dispatch to HF Jobs); `monitor.yml` (Hub API → job summary). |
| `.opencode/` | 25 slash-command specs (`command/hf-*.md`) + 35 workflow skills (`skills/*/SKILL.md`) — prompt library for the whole pipeline. Path-agnostic; no code depends on it. |
| `docs/HF_HUB_IMPROVEMENTS.md`, `SECURITY.md` | 2026-07-30 Hub audit; token-hygiene checklist. |
| `verify_grpo_contract.py`, `test_browsergym_contract.py`, `conftest.py` | CPU-only contract checks, plus the `openenv` stub that lets the two workspace test suites collect here. Together with `.ruff.toml`, the only things runnable in this checkout without HF Jobs / a GPU. |
| `PLAN.md` | The active improvement plan; kept in sync with what has landed on `main`. |

`.gitattributes` at the root is Hugging Face's auto-generated LFS config, merged in from
a Space repo. Leave it alone.

## Running things locally (what actually works here)

```bash
uv run --with ruff ruff check .          # E9 / F821 / F811 only; clean tree
python3 verify_grpo_contract.py          # drives SimpleGuessEnv; skips TRL when absent
uv run --with datasets --with pytest --with pydantic --with httpx \
  pytest test_browsergym_contract.py \
         sakthai-agentic-eval-train/tests/ \
         openenv-custom-training/tests/    # 38 passed
```

Run the three suites in **one** pytest invocation, as CI does. They share `sys.modules`, and
splitting them hides mock leaks between them — `test_eval_hermes_env.py` leaving a MagicMock
at `sys.modules["torch"]` broke `Dataset` construction in the browsergym file, and only
showed up once they ran together.

`test_browsergym_contract.py` mocks `trl` (`sys.modules['trl'] = MagicMock()`) but **not**
`datasets` — plain `pytest test_browsergym_contract.py` fails at import with
`ModuleNotFoundError: No module named 'datasets'` unless `datasets` is installed. It also
does `sys.path.append("openenv-custom-training")`, so it only works from the repo root.

The root `conftest.py` stubs `openenv` (and only when the real package is absent) so the
workspace suites collect on a CPU box; before it existed they failed at import and no CI job
ran them. These tests assert the **contract**, not the implementation — an earlier revision
asserted a bare-string `prompt` and an `env_outputs=` kwarg TRL never passes, and so stayed
green while both were bugs. Keep it that way.

Run all of it before touching anything under `openenv-custom-training/`. Everything else needs
a GPU box; do not attempt to run training here.

## The SFT half (`sakthai-sft-training/`)

This workspace was `beer-sakthai/SakThai-Training` before the 2026-08-21 consolidation.
It produces the base LoRA adapters (`Nanthasit/sakthai-context-{0.5b,1.5b,7b}-tools`) that
the RL half loads and further-trains via GRPO. Its conventions are related to but distinct
from the RL half — do not casually cross-import.

### TRL 0.19 API quirks (SFT scripts only)

The SFT scripts pin TRL 0.19.1 / transformers 5.14.1 / PyTorch 2.13.0 CPU-side; the RL
scripts pin transformers>=5.2.0 + TRL current. These are **incompatible pinsets**; do not
try to unify them in one `requirements.txt`.

- `SFTConfig(processing_class=tokenizer, ...)` — the kwarg is `processing_class`, not `tokenizer`.
- `completion_only_loss=True` masks non-assistant tokens via the model's chat template.
- `hub_strategy="every_save"` protects against HF-Jobs timeouts; set `--timeout` explicitly on
  HF Jobs (the 30-min default is too short for anything past the smoke script).

### The 10-cycle augmentation loop

`cycle-100-v2.py` … `cycle-100-v10.py` are ten immutable snapshots of the same
data-augmentation script — each one produced one released dataset revision
(`Nanthasit/sakthai-combined-v{2..10}`). Do not "refactor them into one file": the whole
point is that the exact bytes that produced each dataset revision are pinned. New rounds
add a `cycle-100-v11.py`; they do not edit older ones. `cycle-workflow-gap-fill.py` is the
gap-fill variant used to build `gap-fill-v8/v8-gap-fill.jsonl` (478 rows).

### Datasets on the Hub, not in the repo

Except for `gap-fill-v8/v8-gap-fill.jsonl` (kept inline for reproducibility), all training
corpora live on the Hub — `Nanthasit/sakthai-combined-v6`, `v7`, `v10` and `v12`
(~2.4k → ~5k rows; **v8, v9 and v11 were never pushed** — `hf stat` reports all
three missing as of 2026-09-16, and `train-sakthai-1.5b-v2.py` loads v8, so that
script cannot run until it is created or repointed) and
`Nanthasit/sakthai-bench-v{1..3}` (155 balanced rows in v3). Do not check dataset payloads
into the repo. The runtime pin dataset `Nanthasit/sakthai-openenv-training` is the source
of truth for cross-half version compatibility.

### Reading benchmark numbers

The SFT-half benchmark table (0.5B / 1.5B / 7B on `sakthai-bench-v3`) lives in
`sakthai-sft-training/README.md`. The RL-half agentic-eval numbers live in
`sakthai-agentic-eval-train/FINDINGS.md`. **Do not restate either set of numbers
elsewhere** — `FINDINGS.md` is the durable record for the RL half and the workspace README
is the record for the SFT half; anything else drifts.

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
  does not exist and crashes at construction. Fixed in `train.py`, `multi_env.py`, and
  `openenv-multi-catalog-training/train_multi_env.py:main()` (last one landed 2026-08-21).
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

A parseable index of these items lives at [`docs/KNOWN_GAPS.yaml`](docs/KNOWN_GAPS.yaml);
the prose below remains authoritative for context, but automation should read the YAML.
When you close one, delete both.

- `coding_env`'s task in both `train_multi_env.py` and `a2a_agent/` is a placeholder
  (`print(17 * 23)`) with a substring check for correctness.
- No catalog Docker image tag in `run_servers.sh` has been verified live.
- `a2a_agent/` has never been executed; the `TaskUpdater` method names were written from the
  published SDK pattern, not a live install.
- The 7B GRPO proof-of-signal run was 40 steps — long enough to show a gradient exists, not
  long enough to improve the model. A real run needs hundreds of steps.
- HF Jobs returns `402 Payment Required` on this account. **Credit is the only blocker** —
  `HF_TOKEN` IS configured as a repo secret and valid: the 2026-09-14 `lighteval` run logs
  `Token is valid (permission: write)` and `Login successful`, then fails with
  `402 ... Pre-paid credit balance is insufficient`. The four HF-Jobs workflows cannot
  dispatch until credit is added, but they no longer go red for it: each routes its
  `hf jobs` call through `.github/scripts/hf-jobs-submit.sh` (added 2026-09-18), which
  reports a 402 as a skip — warning annotation, job summary, exit 0 — and lets every other
  failure through untouched. A red run on one of them therefore means a real problem, not
  an empty balance. `verify-contracts.yml` and `hf-no-cost-checks.yml` run regardless.
- `--env browsergym`'s wrapper (`_BrowserGymTaskEnv` in `openenv-custom-training/train.py`)
  is **written, not run**. `BrowserGymAction(action=...)` and the observation text field
  were inferred, not checked against an installed `browsergym_env`. Confirm both before a
  long run. The server it points at is now upstream `openenv/browsergym_env` (its OpenEnv
  routes verified 2026-09-16); `Nanthasit/browsergym-env` was deleted at some point and
  returns 404, so `browsergym-space/` is a redeploy recipe, not a live deployment.
- Dataset payloads are committed under `augmented-output/` (~1 MB) and
  `safety-quality-fixes/`, against the "datasets live on the Hub, not in the repo" rule
  below. `augmented-output/push-augmented.py` reads `all-augmented.jsonl` from there, so
  removing them is not a pure deletion — it needs that script repointed at the Hub first.
- `sakthai-combined-v8`, `v9` and `v11` do not exist on the Hub (verified 2026-09-16).
  `push-all-to-hub.py` (v8) could not have created it — it failed to parse until
  2026-09-16 — and `push-v9-comprehensive.py` (v9) was evidently never run either.
  `train-sakthai-1.5b-v2.py:88` loads v8 inside a `try/except` that prints
  `v8 unavailable:` and continues, so `train.yml` does not fail on the miss — it just
  silently drops the v8 augmentation shard and trains on v7 alone, which is not what the
  script's "v2" naming implies. Push v8 (via `push-all-to-hub.py` once it is reviewed) or
  repoint that line at an existing revision (`v10` is the latest that exists) before
  spending credit on `train.yml`.
- ~115 ruff style findings (F541 / F401 / F841) remain unaddressed, concentrated in
  `sakthai-sft-training/`. They are out of the lint gate on purpose; see the `.ruff.toml`
  header before widening it.

**Resolved 2026-09-16** (repo-health pass):

- ~~`sakthai-sft-training/push-all-to-hub.py` failed to parse~~ → unterminated module
  docstring closed; it was the only file `compileall` rejected.
- ~~`train.py --env browsergym` scored every episode 0.0 and passed a bare-string
  `prompt`~~ → reward now reads `env.reward` off the instances via a proper wrapper, and
  the dataset is conversational. Both contract tests had been asserting the bugs.
- ~~`a2a_agent/agent_executor.py` carried 13 duplicated methods from a bad merge~~ →
  dead first block removed (423 → 340 lines), live behaviour proven unchanged.
- ~~Two NameErrors in eval scripts (`del m`, missing `import collections`)~~ → fixed;
  both fired partway through a paid GPU job.
- ~~27 workspace tests that no CI job ran~~ → root `conftest.py` stubs `openenv`;
  `verify-contracts.yml` now runs 38 tests plus ruff.
- ~~`monitor.yml` failed every week by design; `train.yml` used a nonexistent runner
  label; `mcp-bench.yml` called `hf` without installing it; `manual.yml` / `stale.yml`
  were unmodified starter templates~~ → fixed, or deleted in `manual.yml`'s case.

**Resolved 2026-08-21** (consolidation PR):

- ~~`train_multi_env.py` passes nonexistent `vllm_server_url`~~ → now uses `vllm_server_host` + `vllm_server_port`.
- ~~`train.py --env browsergym` missing `browsergym_env` in PEP 723 header + requirements.txt~~ → added to both.
