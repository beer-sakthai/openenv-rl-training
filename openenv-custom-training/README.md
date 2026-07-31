# Custom-environment OpenEnv + TRL GRPO training

Companion to the catalog-only workspace at `../openenv-multi-catalog-training/`.
That workspace trains across the 8 `openenv/*` **catalog** environments; this
one designs **custom** environments — the task and reward are yours.

Two environment shapes live here:

| Tier | File | When to use | Isolation |
|------|------|-------------|-----------|
| **A — inline logic** | `env_simple_task.py` | Task is safe plain Python (no exec of model-generated code) | None — runs in the training process |
| **B — sandboxed server** | `agent_tools/` | Model's actions execute arbitrary/untrusted code (terminal, code-exec) | Real OpenEnv server in Docker |

Everything is **design-only**: this machine has no GPU, nothing here has been
run. Take the workspace to a GPU box (Kaggle, HF Jobs, a rented box) and run
it there.

## Layout

```
env_simple_task.py        Tier A: SimpleGuessEnv + reward + dataset (inline, no server)
agent_tools/              Tier B: sandboxed OpenEnv server + client + TRL wrapper
  models.py               Action/Observation (Pydantic)
  server/sandbox_env.py   the actual terminal task logic, runs INSIDE the sandbox
  server/app.py           create_app wiring (concurrency opt-in)
  client.py               EnvClient subclass the wrapper talks to
  wrapper.py              the TRL-facing class passed to environment_factory
  README.md               openenv init / build / run instructions
train.py                  entrypoint: --env {simple,agent_tools}, configurable model & vLLM mode
multi_env.py              Tier A + Tier B combined, TRL-native dict form
                          (environment_factory={"guess": ..., "agent_tools": ...})
requirements.txt          pinned deps (written against openenv 0.4.1 / trl 1.9.2)
```

## The one decision that matters: inline vs. sandboxed

GRPO exploration means the policy **will** eventually emit off-distribution,
weird actions. If a tool method runs whatever string the model produced,
that's arbitrary code execution in your training process. That is exactly the
situation for terminal/code-exec tools → Tier B. If the task never executes
untrusted code, Tier A's plain-Python logic is legitimate and far cheaper to
set up.

## Model size (your "mix" answer)

`--model` is a first-class flag everywhere; the default is the SakThai family's
flagship full checkpoint, `Nanthasit/sakthai-context-1.5b-merged` (Qwen2
1.5B, 32K context, tool-calling-capable chat template). One base covers both
tiers — the same model learns the simple game and the sandboxed tool tasks.

Why not the smaller `sakthai-context-0.5b-merged`? It ships no chat template,
which GRPO needs to format prompts. For a smaller base, pass one via `--model`
(e.g. a Qwen3-0.6B); the swap is the only change needed. Bigger model ⇒ more
VRAM for both the policy and the (colocated) vLLM engine.

## Running elsewhere

```bash
pip install -r requirements.txt

# Tier A — no server needed
python train.py --env simple --vllm-mode colocate

# Tier B — start the sandbox server first (see agent_tools/README.md), then:
python train.py --env agent_tools --vllm-mode colocate

# Both at once
python multi_env.py --vllm-mode colocate
```

- **1 GPU** → `--vllm-mode colocate`. **2+ GPUs** → start
  `CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model <model> --port 8000`, then
  add `--vllm-mode server --vllm-server-url http://localhost:8000`.
- **`max_completion_length` caps tokens across the WHOLE multi-turn episode**
  (every generation + every tool result, summed). Episodes truncated mid-task?
  Raise it before suspecting anything else.
- **Concurrency**: the Tier B server must opt in
  (`SUPPORTS_CONCURRENT_SESSIONS = True`, `max_concurrent_envs` ≥
  `num_generations`). Tier A needs no server at all.
- **Sanity-check rewards first**: run a few manual episodes and confirm a
  capable model scores above random before spending GPU time. Both workspaces'
  envs are callable by hand for this.

## Reward philosophy (applies to both tiers)

Binary (1.0/0.0), judged on **final state** — does the episode succeed — not
on the path taken. GRPO ranks within a group, so what matters is the ordering
a reward induces; outcome-only rewards let the model find strategies you
didn't. In Tier B the *server* evaluates acceptance (it owns the filesystem
state) and reports `reward`/`done` in each observation; the wrapper just
forwards it.

## Version pins & the experimental-API caveat

`openenv` is experimental and its APIs shift. On 2026-07-31 the Tier B stack
was **verified end-to-end against the real `openenv==0.4.1`** (server app
booted under uvicorn, real client sessions drove all three tasks, concurrency
checked) and the **`trl==1.9.2` source was read** to confirm the
`environment_factory` / `GRPOConfig` contract. Two hard requirements surfaced:

- **`transformers>=5.2.0`** — `GRPOTrainer` raises if lower when
  `environment_factory` is used.
- **`jmespath`** — required for tool-response parsing.
- The base model's **chat template must support tool calling**
  (`supports_tool_calling`); `GRPOTrainer` validates this and raises otherwise.
  `Nanthasit/sakthai-context-1.5b-merged` qualifies (its template has a
  `{%- if tools %}` branch) — a base without a tool-calling template is not a
  valid GRPO-with-environment base.

If any surface drifts again, re-check huggingface.co/docs/openenv and
huggingface.co/docs/trl/en/openenv rather than assuming these files still match.
