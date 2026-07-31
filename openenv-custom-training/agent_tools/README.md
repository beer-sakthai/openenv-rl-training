# Tier B: sandboxed agent-tools environment

Use this when the model's actions execute **arbitrary/untrusted code** — i.e.
training against terminal / code-execution tools the way the live agents use
them. GRPO exploration means the policy *will* try off-distribution actions;
this shape keeps that inside an isolated OpenEnv server, not in your training
process.

Three moving pieces (the "two different things called the environment"
confusion, spelled out):

1. **Server-side `SandboxEnvironment`** (`server/sandbox_env.py`) — the real
   terminal logic, running *inside* the sandboxed server. Owns the filesystem
   state, evaluates task acceptance, sets `reward`/`done`.
2. **`AgentToolClient`** (`client.py`) — translates between the wire payload
   and the typed models. Holds a connection to the server.
3. **`AgentToolEnv`** (`wrapper.py`) — the TRL-facing class passed to
   `environment_factory=...`. Exposes named tools (`run_command`,
   `write_file`, `read_file`) with docstrings; its job is to translate tool
   calls into `client.step(...)` and forward `reward`/`done` to TRL.

## Build & run (on the training box)

`openenv init` scaffolds the canonical project (server entrypoint, Dockerfile,
`openenv.yaml`); these files are designed to drop into that scaffold.

```bash
pip install -r requirements.txt
openenv init agent_tools          # generates server/Dockerfile, pyproject, etc.
#   -> copy this directory's models.py, server/sandbox_env.py, server/app.py,
#      client.py into the scaffold, replacing the stubs.
uv run --project agent_tools server   # iterate locally
openenv build                        # build the Docker image
# run it on port 8001 (leaving 8000 free for a colocated vLLM server):
docker run -d -p 8001:8000 --platform linux/amd64 \
    registry.hf.space/<you>/agent_tools:latest

# then, in the training workspace:
AGENT_TOOLS_URL=http://localhost:8001 \
python ../train.py --env agent_tools --vllm-mode colocate
```

### Concurrency (required, not optional)

GRPO opens one WebSocket per generation in the batch. The server must opt in:

- `SUPPORTS_CONCURRENT_SESSIONS = True` in `server/sandbox_env.py` (done).
- `max_concurrent_envs=<N>` to `create_app(...)` in `server/app.py`, where
  `N >= num_generations` (`per_device_train_batch_size ×
  gradient_accumulation_steps`). Default 8; raise if you raise the batch.

### API contract — verified against openenv 0.4.1 on 2026-07-31

Everything below was checked against the installed 0.4.1 source, and the
server+client were run **live end-to-end** (uvicorn + real WebSocket
sessions):

- `create_app(EnvClass, Action, Observation, env_name=..., max_concurrent_envs=...)`
  — confirmed; `env` is a zero-arg factory, so passing the class directly works.
- `SUPPORTS_CONCURRENT_SESSIONS` must be a **class attribute** on the
  `Environment` subclass (the server reads it off the class and raises
  `ConcurrencyConfigurationError` if `max_concurrent_envs>1` without it).
- `Environment.reset(seed=None, episode_id=None, **kwargs)` / `step(action,
  timeout_s=None, **kwargs)` — the server filters kwargs against your
  signature, so the permissive overrides here (`task_key` via kwargs) are safe.
- Client: `EnvClient` lives at `openenv.core` (re-exported); a concrete client
  subclasses `EnvClient[Action, Obs, State]` and implements `_step_payload`,
  `_parse_result`, `_parse_state`. `reset()`/`step()` return a `StepResult`;
  the observation lives at `.observation` (reward/done mirrored top-level).
- `.sync()` then `__enter__()` is the real session-open pattern.

The one remaining runtime unknown is packaging (`openenv init` scaffold +
`openenv build` + Docker), which this machine can't run (no working Docker
daemon) — `server/Dockerfile` comes from the scaffold.

## Security model — what isolates the exec, and what does not

The sandbox boundary is the **container**, not string blocklists:

- **Container**: non-root user, no network, small/empty filesystem, `--pids`
  limit. Recommended Dockerfile directives (edit the scaffold's Dockerfile):
  `USER nobody`, no `apt`/`pip` in the image beyond the env package, run with
  `docker run --network none --pids-limit 128`. The reference material's
  point is that blocklisting `rm -rf` etc. is a false sense of security — the
  policy will find a spelling you didn't block. Isolation, not lists.
- **Per-step guards in code** (what this repo adds on top): command timeout
  (10s), stdout/stderr output caps, cwd confined to a per-episode scratch
  dir, and an env stripped of secrets. `write_file`/`read_file` refuse paths
  that `realpath` outside the scratch dir.

An episode that runs long is fine (a step is capped); an episode that never
converges is cut off by `STEP_LIMIT` with a clean 0.0.

## Reward

Binary, outcome-based, judged by the **server** (it owns the state):
after every step the server re-evaluates the task's acceptance predicate
against the current filesystem and returns `reward=1.0, done=True` the moment
it passes. The wrapper forwards it untouched. `train.py --env agent_tools`
uses exactly this `reward_func`.
