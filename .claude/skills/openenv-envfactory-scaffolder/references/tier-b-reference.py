"""Tier B reference — the sandboxed-server pattern in one file for reading.

Tier B is ACTUALLY three (or four) files in the real layout, mirroring
agent_tools/:

    openenv-custom-training/<name>_tools/
        wrapper.py              # TRL-facing class, thin, forwards to client
        server/sandbox_env.py   # task logic, owns state, scores each step
        models.py               # Action / Observation dataclasses
        client.py               # HTTP shim (usually copy-adapt of agent_tools/client.py)

This file inlines all three so a reader can see the whole contract at once.
When scaffolding, split it back into the three files.

Use Tier B when — and ONLY when — some tool method executes strings the
model produced (shell commands, Python code, SQL, regex fed to re.sub, a
template string rendered into a filesystem path). The isolation is the
CONTAINER, not a string blocklist. There is deliberately no command
blocklist in _run_command below; the policy will find a spelling you
didn't block.

Task in this example: a sandboxed shell. Three named tasks selected by a
`task` dataset column. Each task has a prompt (what the model sees) and an
acceptance predicate (what makes reward=1.0). Server-side.
"""

# ============================================================================
# server/sandbox_env.py — task logic (server-side, runs INSIDE the container)
# ============================================================================

import os
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# The dataclasses defined in models.py further down.
# from ..models import AgentToolAction, AgentToolObservation

# --- Per-step guards --------------------------------------------------------
# Every exec path (_run_command, _write_file, _read_file) MUST enforce these.
# No blocklist substitutes for them.
STEP_LIMIT = 15          # episode cap — a non-converging rollout terminates
CMD_TIMEOUT_S = 10       # per-command timeout; a hung cmd blocks the slot
MAX_STDOUT = 4000        # cap output — noise the model can't use starves it
MAX_STDERR = 2000

# --- Task registry: what the model sees + how to score it -------------------
# One class handling multiple named tasks keyed by a `task` column is almost
# always the right shape. The prompt is what the model reads; the acceptance
# predicate is evaluated after every step against the current filesystem.

TASKS = {
    "answer_file": {
        "prompt": (
            "Create a file named answer.txt that contains exactly the number 42 "
            "(no other text), then run `cat answer.txt` to confirm. Use "
            "write_file and run_command."
        ),
        "seed": None,
    },
    "count_lines": {
        "prompt": (
            "Create a file named lines.txt with exactly three lines: alpha, "
            "beta, gamma (one word per line). Then write the number of lines "
            "in lines.txt to a file named out.txt. Use write_file and "
            "run_command."
        ),
        "seed": None,
    },
    "upper_copy": {
        "prompt": (
            "words.txt contains words, one per line. Create a new file "
            "upper.txt with every word in ALL CAPS, one per line. Use "
            "read_file and write_file."
        ),
        "seed": "apple\nbanana\ncherry\n",
    },
}


def _accepts(task_key: str, workdir: Path) -> bool:
    """Acceptance predicate for the given task, evaluated against the current
    workdir state. Returns True the first time the task's success condition
    holds; the server then sets reward=1.0 and ends the episode.
    """
    if task_key == "answer_file":
        p = workdir / "answer.txt"
        return p.exists() and p.read_text().strip() == "42"
    if task_key == "count_lines":
        p = workdir / "out.txt"
        if not p.exists():
            return False
        return p.read_text().strip() == "3"
    if task_key == "upper_copy":
        src, dst = workdir / "words.txt", workdir / "upper.txt"
        if not (src.exists() and dst.exists()):
            return False
        return dst.read_text() == src.read_text().upper()
    return False


class SandboxEnvironment(Environment):
    """One sandbox session == one episode. Constructed FRESH per session by
    the server (never pass an instance to create_app).
    """

    # CLASS attribute, not module-level. The server does
    # getattr(env_cls, "SUPPORTS_CONCURRENT_SESSIONS", False) and refuses
    # max_concurrent_envs > 1 without it. This is the single most common
    # concurrency bug when copy-adapting from another env.
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        # Fresh scratch dir per session so one session's artefacts never leak
        # into another's grading.
        self._workdir = Path(tempfile.mkdtemp(prefix="opentrain-"))
        self._task_key = os.environ.get("DEFAULT_TASK_KEY", "answer_file")
        self._reward = 0.0
        self._done = False

    @property
    def state(self) -> State:
        return self._state

    def reset(self, seed=None, episode_id=None, **kwargs):
        """The server calls this via reset_async. task_key travels via kwargs;
        seed/episode_id are accepted and intentionally unused — a fresh
        scratch dir per reset already isolates episodes.
        """
        task_key = kwargs.get(
            "task_key", os.environ.get("DEFAULT_TASK_KEY", "answer_file")
        )
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_key = task_key
        self._reward = 0.0
        self._done = False
        self._workdir = Path(tempfile.mkdtemp(prefix="opentrain-"))
        if (seed := TASKS[task_key].get("seed")) is not None:
            (self._workdir / "words.txt").write_text(seed)
        prompt = TASKS[task_key]["prompt"]
        # AgentToolObservation is a dataclass with .result, .success, .done,
        # .reward — see models.py further down.
        return AgentToolObservation(
            result=prompt, success=True, done=False, reward=0.0
        )

    def step(self, action, timeout_s=None, **kwargs):
        """Server calls this per tool call. Action is a validated dataclass
        with .tool ("run_command"|"write_file"|"read_file") and .args (dict).
        """
        if self._done:
            return AgentToolObservation(
                result="Episode already ended.", success=False, done=True,
                reward=self._reward,
            )
        self._state.step_count += 1
        try:
            if action.tool == "run_command":
                result, success = self._run_command(action.args.get("command", ""))
            elif action.tool == "write_file":
                result, success = self._write_file(
                    action.args.get("path", ""), action.args.get("content", "")
                )
            elif action.tool == "read_file":
                result, success = self._read_file(action.args.get("path", ""))
            else:
                raise ValueError(f"Unknown tool: {action.tool!r}")
        except Exception as exc:
            # Exceptions come back to the model as the tool result string.
            result, success = f"error: {exc}", False

        if _accepts(self._task_key, self._workdir):
            self._reward = 1.0
            self._done = True
            result = f"{result}\n\nTASK COMPLETE."
        elif self._state.step_count >= STEP_LIMIT:
            self._done = True  # clean 0.0, not an infinite loop
            result = f"{result}\n\nSTEP LIMIT REACHED ({STEP_LIMIT})."
        return AgentToolObservation(
            result=result, success=success, done=self._done, reward=self._reward
        )

    # -- confined exec -------------------------------------------------------

    def _confine(self, rel_or_abs: str) -> Path:
        """Resolve a path inside the workdir, refusing escapes.

        realpath check — not a substring filter on the string. Blocks
        ../../etc/passwd and any symlink trick, without pattern-matching.
        """
        p = (self._workdir / rel_or_abs).resolve()
        if not p.is_relative_to(self._workdir.resolve()):
            raise ValueError(f"path escapes the sandbox workspace: {rel_or_abs!r}")
        return p

    def _run_command(self, command: str) -> tuple[str, bool]:
        if not command.strip():
            raise ValueError("empty command")
        # Env stripped to the bare minimum. Do NOT inherit the training
        # process's env — anything leaked in becomes an attack surface.
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG")}
        proc = subprocess.run(
            ["bash", "-l"], input=command,
            cwd=self._workdir, env=env, capture_output=True, text=True,
            timeout=CMD_TIMEOUT_S,
        )
        stdout = proc.stdout[:MAX_STDOUT] + ("…" if len(proc.stdout) > MAX_STDOUT else "")
        stderr = proc.stderr[:MAX_STDERR] + ("…" if len(proc.stderr) > MAX_STDERR else "")
        out = f"exit_code: {proc.returncode}\nstdout:\n{stdout or '(none)'}\nstderr:\n{stderr or '(none)'}"
        return out, proc.returncode == 0

    def _write_file(self, path: str, content: str) -> tuple[str, bool]:
        target = self._confine(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"wrote {len(content)} bytes to {path}", True

    def _read_file(self, path: str) -> tuple[str, bool]:
        target = self._confine(path)
        if not target.exists():
            return f"error: no such file: {path}", False
        text = target.read_text()
        return text[:MAX_STDOUT] + ("…" if len(text) > MAX_STDOUT else ""), True


# ============================================================================
# models.py — dataclasses used across the wire
# ============================================================================

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class AgentToolAction:
    """One tool call from the model. `tool` is a Literal so pydantic /
    dataclass validation rejects unknown tool names at the boundary."""
    tool: Literal["run_command", "write_file", "read_file"]
    args: dict[str, Any]


@dataclass
class AgentToolObservation:
    """One tool result returned to the model. The `reward`/`done` fields
    make the SERVER the source of truth — the TRL-side wrapper just
    forwards them without re-judging."""
    result: str
    success: bool
    done: bool
    reward: float


# ============================================================================
# wrapper.py — TRL-facing class (runs in the training process, NOT sandboxed)
# ============================================================================

# from .client import AgentToolClient  # HTTP shim, one method per named tool

AGENT_TOOLS_URL = os.environ.get("AGENT_TOOLS_URL", "http://localhost:8001")

# Task keys must match the server's TASKS registry. Duplicated here so the
# dataset can enumerate them without importing the server module.
TASK_KEYS = ("answer_file", "count_lines", "upper_copy")
TASK_PROMPTS = {
    "answer_file": "You have a sandboxed shell. Complete the stated task with run_command / write_file.",
    "count_lines": "You have a sandboxed shell. Complete the stated task with run_command / write_file.",
    "upper_copy": "You have a sandboxed shell. Complete the stated task with read_file / write_file.",
}


class AgentToolEnv:
    """The class GRPOTrainer instantiates once per generation slot.

    Thin by design — it holds a client, translates named tool calls into
    client.step(...), and forwards the server's reward/done to TRL. All the
    task logic and scoring lives on the server, not here.
    """

    def __init__(self):
        self._client = None
        self.reward = 0.0
        self.done = False

    def _open(self):
        # One session per episode, so concurrency scales with the batch and
        # stale episodes release their slot on re-route. .sync() yields the
        # synchronous wrapper that step()/reset()/close() live on.
        client = AgentToolClient(base_url=AGENT_TOOLS_URL)
        synced = client.sync()
        synced.__enter__()
        return synced

    def _close(self):
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def reset(self, **kwargs) -> str | None:
        # `task` comes from the dataset column and selects the server-side
        # task; the server returns that task's prompt as the first observation.
        self._close()
        self.reward = 0.0
        self.done = False
        self._client = self._open()
        task_key = kwargs.get("task", "answer_file")
        obs = self._client.reset(task_key=task_key).observation
        return obs.result

    # -- tools (docstrings ARE the schema the model reads) --------------------

    def run_command(self, command: str) -> str:
        """Run a shell command in the sandbox and see stdout/stderr/exit code.

        The command runs in an isolated workspace (no network, time-limited,
        output-capped). Use it to inspect or transform files, then confirm
        your result before finishing.

        Args:
            command: the shell command line to execute.
        """
        return self._forward(self._client.run_command(command))

    def write_file(self, path: str, content: str) -> str:
        """Create or overwrite a file in the sandbox workspace.

        Args:
            path: file path relative to the workspace root.
            content: full file contents to write.
        """
        return self._forward(self._client.write_file(path, content))

    def read_file(self, path: str) -> str:
        """Read a file from the sandbox workspace.

        Args:
            path: file path relative to the workspace root.
        """
        return self._forward(self._client.read_file(path))

    def _forward(self, step_result) -> str:
        # Client helpers return a StepResult; the observation (with the
        # server's result/reward/done fields) lives at .observation. Copy
        # reward/done onto self so reward_func can read them after the
        # episode.
        obs = getattr(step_result, "observation", step_result)
        self.reward = obs.reward
        self.done = obs.done
        return obs.result


def reward_func(environments, **kwargs) -> list[float]:
    """Binary, outcome-based: forward the server's 1.0/0.0 verdict.

    The server judges the FINAL filesystem state against the acceptance
    predicate — the model is free to use any sequence of commands that
    gets there.
    """
    return [env.reward for env in environments]


def build_dataset(n_episodes: int = 64):
    """Rows carry a `task` column the server uses to select the acceptance
    predicate; the model sees only `prompt` (which names the task).
    Deterministic: task_key = TASK_KEYS[i % len(TASK_KEYS)] so successive
    episodes rotate through the registry evenly.
    """
    from datasets import Dataset

    rows, tasks = [], []
    for i in range(n_episodes):
        task_key = TASK_KEYS[i % len(TASK_KEYS)]
        rows.append([{"role": "user", "content": TASK_PROMPTS[task_key]}])
        tasks.append(task_key)
    return Dataset.from_dict({"prompt": rows, "task": tasks})


# --- Module-level exports (in wrapper.py, not server-side) ------------------
ENVIRONMENT_FACTORY = AgentToolEnv
REWARD_FUNCS = reward_func
TRAIN_DATASET = build_dataset
DEFAULT_MODEL = "Nanthasit/sakthai-context-7b-tools"
