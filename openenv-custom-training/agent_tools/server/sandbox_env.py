"""Server-side environment: the terminal task logic that runs INSIDE the
sandboxed OpenEnv server.

This is the "actual task/game logic" half of Tier B (see agent_tools/README.md
for the three-piece split). It owns the filesystem state, executes commands,
and — because it owns the state — evaluates each task's acceptance predicate
and reports `reward`/`done` back in every observation. The TRL-facing wrapper
never scores anything itself; it just forwards these fields.

Isolation comes from the container this runs in (non-root, no network,
pids-limited) plus the per-step guards in `_run_command`/`_write_file`/
`_read_file` (timeout, output caps, confined cwd, stripped env, realpath
checks). There is deliberately NO command string-blocklist — a blocklist is a
false sense of security; the policy will find a spelling you didn't block.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from ..models import AgentToolAction, AgentToolObservation

# --- concurrency opt-in ----------------------------------------------------
# Must be a CLASS attribute (the base Environment declares it as such and the
# server reads it via getattr(env_cls, ...)). True allows up to
# max_concurrent_envs simultaneous sessions — training opens one WebSocket per
# generation in the batch, so this must be True (and create_app must get
# max_concurrent_envs >= num_generations).

STEP_LIMIT = 15
CMD_TIMEOUT_S = 10
MAX_STDOUT = 4000
MAX_STDERR = 2000

# --- task registry: prompt the model sees + acceptance predicate -----------
# Acceptance is evaluated by the server against the current filesystem after
# every step; the first time it passes, reward=1.0 and the episode ends.

TASKS = {
    "answer_file": {
        "prompt": (
            "Create a file named answer.txt that contains exactly the number 42 "
            "(no other text), then run `cat answer.txt` to confirm. Use "
            "write_file and run_command."
        ),
        "seed": None,  # no pre-existing files
    },
    "count_lines": {
        "prompt": (
            "Create a file named lines.txt with exactly three lines: alpha, beta, "
            "gamma (one word per line). Then write the number of lines in lines.txt "
            "to a file named out.txt. Use write_file and run_command. The grading "
            "checks out.txt contains the correct line count."
        ),
        "seed": None,
    },
    "upper_copy": {
        "prompt": (
            "words.txt contains words, one per line. Create a new file upper.txt "
            "with every word in words.txt in ALL CAPS, one per line. Use read_file "
            "and write_file."
        ),
        "seed": "apple\nbanana\ncherry\n",
    },
}


def _accepts(task_key: str, workdir: Path) -> bool:
    if task_key == "answer_file":
        p = workdir / "answer.txt"
        return p.exists() and p.read_text().strip() == "42"
    if task_key == "count_lines":
        p = workdir / "out.txt"
        if not p.exists():
            return False
        out = p.read_text().strip()
        return out == "3"
    if task_key == "upper_copy":
        src, dst = workdir / "words.txt", workdir / "upper.txt"
        if not (src.exists() and dst.exists()):
            return False
        return dst.read_text() == src.read_text().upper()
    return False


class SandboxEnvironment(Environment):
    """One sandbox session == one episode. Constructed fresh per session by
    the server (never pass an instance to create_app)."""

    # Class attribute, not module-level — the server reads it off the class
    # (see interfaces.py) and refuses max_concurrent_envs>1 without it.
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._workdir = Path(tempfile.mkdtemp(prefix="opentrain-"))
        self._task_key = "answer_file"
        self._reward = 0.0
        self._done = False

    @property
    def state(self) -> State:
        return self._state

    # -- Gym-ish lifecycle ----------------------------------------------------
    # Signatures must match the Environment base: the server calls
    # reset_async(seed=..., episode_id=..., **kwargs) -> reset(...) and
    # step_async(action, timeout_s=...) -> step(...). task_key travels via
    # kwargs; seed/episode_id are accepted and intentionally unused (fresh
    # scratch dir per reset already isolates episodes).

    def reset(self, seed=None, episode_id=None, **kwargs) -> AgentToolObservation:
        task_key = kwargs.get("task_key", "answer_file")
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_key = task_key
        self._reward = 0.0
        self._done = False
        # Fresh scratch dir per episode so one session's artifacts never leak
        # into another's grading.
        self._workdir = Path(tempfile.mkdtemp(prefix="opentrain-"))
        if (seed := TASKS[task_key].get("seed")) is not None:
            (self._workdir / "words.txt").write_text(seed)
        prompt = TASKS[task_key]["prompt"]
        return AgentToolObservation(
            result=prompt, success=True, done=False, reward=0.0
        )

    def step(self, action: AgentToolAction, timeout_s: float | None = None, **kwargs) -> AgentToolObservation:
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
            else:  # unreachable given the Action Literal, but be safe
                raise ValueError(f"Unknown tool: {action.tool!r}")
        except Exception as exc:  # fed back to the model as the tool result
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

    # -- tool implementations (the sandboxed exec) ----------------------------

    def _confine(self, rel_or_abs: str) -> Path:
        """Resolve a path inside the workdir, refusing escapes."""
        p = (self._workdir / rel_or_abs).resolve()
        if not p.is_relative_to(self._workdir.resolve()):
            raise ValueError(f"path escapes the sandbox workspace: {rel_or_abs!r}")
        return p

    def _run_command(self, command: str) -> tuple[str, bool]:
        if not command.strip():
            raise ValueError("empty command")
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG")}
        proc = subprocess.run(
            ["bash", "-lc", command],
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
