"""Contract test for the Tier B OpenEnv server entrypoint (agent_tools/server/app.py).

app.py wires create_app(EnvironmentClass, ActionType, ObservationType, ...) at
import time -- passing the wrong class, or an instance instead of a class (its
own module docstring warns against exactly that), breaks every Tier B training
run with no local way to notice until a GPU job starts against it. This locks
the wiring in.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def _fresh_import_app():
    sys.modules.pop("agent_tools.server.app", None)
    import agent_tools.server.app as app_module
    return app_module


def test_app_wires_sandbox_environment_with_correct_types(monkeypatch):
    monkeypatch.delenv("MAX_CONCURRENT_ENVS", raising=False)
    from openenv.core.env_server import create_app
    create_app.reset_mock()

    _fresh_import_app()

    from agent_tools.models import AgentToolAction, AgentToolObservation
    from agent_tools.server.sandbox_env import SandboxEnvironment

    create_app.assert_called_once_with(
        SandboxEnvironment,
        AgentToolAction,
        AgentToolObservation,
        env_name="agent_tools",
        max_concurrent_envs=8,
    )


def test_app_reads_max_concurrent_envs_from_env(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_ENVS", "16")
    from openenv.core.env_server import create_app
    create_app.reset_mock()

    _fresh_import_app()

    assert create_app.call_args.kwargs["max_concurrent_envs"] == 16
