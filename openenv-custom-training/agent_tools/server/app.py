"""OpenEnv server entrypoint. Pass the CLASS (or a zero-arg factory), never an
instance — the server constructs one environment per session.

Run locally (iteration):   uv run --project <scaffold> server
As a container (training): openenv build, then docker run (see README.md).
"""

import os

from openenv.core.env_server import create_app

from ..models import AgentToolAction, AgentToolObservation
from .sandbox_env import SandboxEnvironment

# Concurrency opt-in: must be >= num_generations when training.
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT_ENVS", "8"))

app = create_app(
    SandboxEnvironment,
    AgentToolAction,
    AgentToolObservation,
    env_name="agent_tools",
    max_concurrent_envs=MAX_CONCURRENT,
)
