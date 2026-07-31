"""TRL-facing wrapper for the sandboxed agent-tools environment.

This is the class passed to `GRPOTrainer(environment_factory=...)`. Its job is
thin: hold a client, translate named tool calls into `client.step(...)`, and
forward the server's `reward`/`done` to TRL. The server does all scoring.

Why a real server here and not inline logic: the tools execute arbitrary
code (`run_command` shells out to whatever the model produced). During GRPO
exploration the policy WILL emit off-distribution commands; those must run in
the sandboxed server, never in the training process. See README.md.
"""

import os

from .client import AgentToolClient

AGENT_TOOLS_URL = os.environ.get("AGENT_TOOLS_URL", "http://localhost:8001")
AGENT_TOOLS_IMAGE = os.environ.get("AGENT_TOOLS_IMAGE")  # optional: connect via docker image

# Task keys must match the server's TASKS registry (server/sandbox_env.py).
TASK_KEYS = ("answer_file", "count_lines", "upper_copy")
TASK_PROMPTS = {
    "answer_file": "You have a sandboxed shell. Complete the stated task with run_command / write_file.",
    "count_lines": "You have a sandboxed shell. Complete the stated task with run_command / write_file.",
    "upper_copy": "You have a sandboxed shell. Complete the stated task with read_file / write_file.",
}


class AgentToolEnv:
    """The class GRPOTrainer instantiates once per generation slot."""

    def __init__(self):
        self._client = None
        self.reward = 0.0
        self.done = False

    def _close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                try:
                    self._client.__exit__(None, None, None)
                except Exception:
                    pass
            self._client = None

    def _open(self) -> AgentToolClient:
        # Session per episode, so concurrency scales with the batch and stale
        # episodes release their slot on re-route (multi-environment pattern).
        # `.sync()` yields the synchronous wrapper that step()/reset()/close()
        # live on — same pattern as the catalog workspace's clients.
        if AGENT_TOOLS_IMAGE:
            client = AgentToolClient.from_docker_image(AGENT_TOOLS_IMAGE)
        else:
            client = AgentToolClient(base_url=AGENT_TOOLS_URL)
        synced = client.sync()
        synced.__enter__()
        return synced

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

    # -- tools ---------------------------------------------------------------
    # Every public method other than reset becomes a callable tool; the
    # docstring IS the schema the model reads.

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

    # -- plumbing ------------------------------------------------------------

    def _forward(self, step_result) -> str:
        # Client helpers return a StepResult; the observation (with the
        # server's result/reward/done fields) lives at `.observation`.
        obs = getattr(step_result, "observation", step_result)
        self.reward = obs.reward
        self.done = obs.done
        return obs.result


def reward_func(environments, **kwargs) -> list[float]:
    """Binary, outcome-based: forward the server's 1.0/0.0 verdict.

    The server judges the FINAL filesystem state against the task's acceptance
    predicate — the model is free to use any sequence of commands that gets
    there.
    """
    return [env.reward for env in environments]


def build_dataset(n_episodes: int = 64) -> list:
    """Rows carry a `task` column the server uses to select the acceptance
    predicate; the model sees only `prompt` (which names the task)."""
    from datasets import Dataset

    rows, tasks = [], []
    for i in range(n_episodes):
        task_key = TASK_KEYS[i % len(TASK_KEYS)]
        rows.append([{"role": "user", "content": TASK_PROMPTS[task_key]}])
        tasks.append(task_key)
    return Dataset.from_dict({"prompt": rows, "task": tasks})


# Reference point for train.py's dispatch table.
ENVIRONMENT_FACTORY = AgentToolEnv
REWARD_FUNCS = reward_func
TRAIN_DATASET = build_dataset
DEFAULT_MODEL = "Nanthasit/sakthai-context-1.5b-merged"
