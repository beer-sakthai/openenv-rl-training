"""Wire types for the sandboxed agent-tools environment.

Action is a discriminated single type: the wrapper's named tool methods
(`run_command`, `write_file`, `read_file`) all collapse into one
`AgentToolAction` whose `tool` field disambiguates, so the server's `step()`
has one dispatcher instead of one Action subclass per tool.
"""

from typing import Any, Literal

from pydantic import Field

from openenv.core.env_server.types import Action, Observation

TOOL_NAME = Literal["run_command", "write_file", "read_file"]


class AgentToolAction(Action):
    tool: TOOL_NAME = Field(..., description="Which sandbox tool to invoke.")
    args: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Tool arguments. run_command: {'command': str}. "
            "write_file: {'path': str, 'content': str}. "
            "read_file: {'path': str}."
        ),
    )


class AgentToolObservation(Observation):
    result: str = Field(..., description="Human-readable result fed back to the model.")
    success: bool = Field(..., description="Whether the tool call succeeded.")
    done: bool = Field(..., description="Whether the task is complete.")
    reward: float = Field(0.0, description="Episode reward so far; 1.0 once the task passes acceptance.")
