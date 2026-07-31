"""Client for the sandboxed agent-tools server.

Mirrors the canonical `openenv init` template client: subclasses
`EnvClient[Action, Observation, State]` and implements the three abstract
methods (`_step_payload`, `_parse_result`, `_parse_state`); the base class
handles the WebSocket protocol, `.sync()` and context-manager semantics.

The base `reset()`/`step()` return a `StepResult` — the observation lives at
`.observation` (and reward/done are mirrored at the top level). The TRL-facing
wrapper (`wrapper.py`) unwraps `.observation`.
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import AgentToolAction, AgentToolObservation


class AgentToolClient(EnvClient[AgentToolAction, AgentToolObservation, State]):
    """Client for the sandboxed agent-tools environment.

    Each instance holds its own dedicated environment session on the server.
    Open a session with the pattern the training wrapper uses:

        client = AgentToolClient(base_url="http://localhost:8001").sync()
        client.__enter__()
        obs = client.reset(task_key="answer_file").observation
    """

    def _step_payload(self, action: AgentToolAction) -> Dict:
        """Convert an AgentToolAction to the JSON payload for the step message."""
        return {"tool": action.tool, "args": action.args}

    def _parse_result(self, payload: Dict) -> StepResult[AgentToolObservation]:
        """Parse the server response into a StepResult[AgentToolObservation]."""
        obs_data = payload.get("observation", {})
        observation = AgentToolObservation(
            result=obs_data.get("result", ""),
            success=obs_data.get("success", False),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward", 0.0)),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse the server state response into a State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

    # -- convenience helpers (return StepResult; wrapper unwraps .observation) --

    def run_command(self, command: str) -> StepResult[AgentToolObservation]:
        return self.step(AgentToolAction(tool="run_command", args={"command": command}))

    def write_file(self, path: str, content: str) -> StepResult[AgentToolObservation]:
        return self.step(AgentToolAction(tool="write_file", args={"path": path, "content": content}))

    def read_file(self, path: str) -> StepResult[AgentToolObservation]:
        return self.step(AgentToolAction(tool="read_file", args={"path": path}))
