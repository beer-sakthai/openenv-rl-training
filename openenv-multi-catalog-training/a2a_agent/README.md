# OpenEnv catalog agent (A2A)

Wraps all 8 `openenv/*` catalog environments (echo, sudoku, coding, chat,
atari, openspiel, repl, sumo) as skills on an [A2A protocol](https://a2a-protocol.org/)
server, so any A2A-speaking agent can drive them directly — independent of
the GRPO training in `../train_multi_env.py`. One A2A task = one OpenEnv
episode.

## Status: written, not run

Nothing in this directory has been executed. This container has no
network-verified `a2a-sdk` install and no way to start/curl a server here.
Before trusting this:

1. `pip install -r requirements.txt`
2. Diff `agent_executor.py`/`server.py` against your installed SDK version
   (`python -c "import a2a; print(a2a.__version__)"`) and the SDK's own
   `helloworld` sample — `TaskUpdater`'s method names (`submit`,
   `start_work`, `update_status`, `failed`, `cancel`) were written from
   memory of the published pattern, not confirmed live.

## Usage (once verified)

```bash
cd ..
bash run_servers.sh                 # starts all 8 env containers
cd a2a_agent
pip install -r requirements.txt
python server.py --port 9000        # serves agent card at :9000/.well-known/agent.json
python client_smoke_test.py --base-url http://localhost:9000/
```

## Protocol shape

- **First message of a task** must carry `metadata={"env": "<name>"}` —
  this resets that environment and returns the initial observation, task
  state `input_required`.
- **Every following message** in the same `task_id` is one step's action:
  - Text-native skills (echo, sudoku, coding, repl) take the action as
    plain text (the message body).
  - `chat` also takes plain text, but it's tokenized with the policy
    model's own tokenizer (`MODEL_NAME` env var, default `Qwen/Qwen3-0.6B`
    — must match whatever model is actually meant to play this skill)
    before being sent as `ChatAction(tokens=...)`, since `chat_env`'s real
    action schema is raw tokens, not text. This must stay consistent with
    `../train_multi_env.py`, which does the same bridging.
  - Int-valued skills (atari, openspiel, sumo) take a `DataPart` with a
    JSON object: `{"action_id": N}` for atari/openspiel, `{"phase_id": N}`
    for sumo.
- Task moves to `completed` (with the final reward folded into the
  response text) once the environment reports `done`; otherwise stays
  `input_required` for the next action.

## Known gaps

- `atari`'s observation is a RAM-byte text dump, not real pixels — same
  vision-mismatch caveat as the training script next door.
- `coding`'s reward check is the same placeholder task (`17 * 23`) as
  `../train_multi_env.py` — replace before using for anything real.
- No auth/rate-limiting — this exposes 8 sandboxed-ish environments (one,
  `coding`, executes arbitrary code) over plain HTTP with no access
  control. Don't expose this port publicly as-is.
