"""Make the workspace test suites runnable on a plain CPU checkout.

Both `openenv-custom-training/tests/` and `sakthai-agentic-eval-train/tests/`
were written to run on a box that already had the training stack installed, so
in a clean checkout they failed at collection and no CI job ran them:

    openenv-custom-training/tests/  -> ModuleNotFoundError: No module named 'openenv'
    sakthai-agentic-eval-train/tests/test_eval_bench_peft.py
        -> ModuleNotFoundError: No module named 'huggingface_hub.utils'

Two different causes, handled separately below.

This file stubs ONLY what is genuinely absent. If the real `openenv` is
installed the stub is skipped entirely, so the tests keep their value on a GPU
box and merely become runnable everywhere else.
"""

import sys
import types
from unittest.mock import MagicMock

# --- 1. Import what is really available, before any test module stubs it -----
# test_eval_bench_peft.py mocks a module only `if mod not in sys.modules`.
# `huggingface_hub` ships as a `datasets` dependency, so it is importable — it
# was simply not yet imported when that guard ran, so the guard replaced a real
# package with a bare MagicMock. A MagicMock is not a package, so the module
# under test then failed on `from huggingface_hub.utils import ...`. Importing
# it here first makes the guard do the right thing.
for _name in ("huggingface_hub", "datasets"):
    try:
        __import__(_name)
    except ImportError:
        pass


# --- 2. Stub `openenv` when it is not installed ------------------------------
def _install_openenv_stub():
    """Register an `openenv` package shaped like the surfaces this repo uses.

    `Action`/`Observation` must be real pydantic models: `agent_tools/models.py`
    subclasses them with `Field(...)` defaults, which a MagicMock cannot carry.
    Everything else only needs to be constructible and subclassable.
    """
    try:
        from pydantic import BaseModel
    except ImportError:  # pragma: no cover - pydantic is in the CI invocation
        BaseModel = object

    class Action(BaseModel):
        pass

    class Observation(BaseModel):
        model_config = {"extra": "allow"}

    class State:
        def __init__(self, episode_id=None, step_count=0, **kwargs):
            self.episode_id = episode_id
            self.step_count = step_count
            for key, value in kwargs.items():
                setattr(self, key, value)

    class Environment:
        SUPPORTS_CONCURRENT_SESSIONS = False

    class StepResult:
        """Subscriptable too: client.py annotates `StepResult[Observation]`."""

        def __class_getitem__(cls, item):
            return cls

        def __init__(self, observation=None, reward=None, done=False, metadata=None):
            self.observation = observation
            self.reward = reward
            self.done = done
            self.metadata = metadata

    class EnvClient:
        """Subscriptable so `EnvClient[Action, Observation, State]` resolves."""

        def __class_getitem__(cls, item):
            return cls

        def __init__(self, base_url=None, **kwargs):
            self.base_url = base_url

    from enum import Enum
    import inspect

    class ToolErrorType(str, Enum):
        TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
        INVALID_ARGS = "INVALID_ARGS"
        EXECUTION_ERROR = "EXECUTION_ERROR"
        TRANSPORT_ERROR = "TRANSPORT_ERROR"
        TIMEOUT = "TIMEOUT"

    class ToolError(BaseModel):
        error_type: ToolErrorType
        message: str = ""

    class Tool(BaseModel):
        name: str
        description: str = ""
        input_schema: dict = {}

    class ListToolsAction(Action):
        pass

    class ListToolsObservation(Observation):
        tools: list[Tool] = []

    class CallToolAction(Action):
        tool_name: str
        arguments: dict = {}

    class CallToolObservation(Observation):
        tool_name: str = ""
        result: object = None
        error: ToolError | None = None

    class MCPEnvironment(Environment):
        """Stub matching openenv.core.env_server.mcp_environment.MCPEnvironment."""

        def __init__(self, mcp_server=None):
            self.mcp = mcp_server
            self._tools_registry = getattr(mcp_server, "_tools_registry", {})

        def step(self, action, timeout_s=None, **kwargs):
            if isinstance(action, ListToolsAction):
                tools = [
                    Tool(
                        name=name,
                        description=inspect.getdoc(fn) or "",
                        input_schema={},
                    )
                    for name, fn in self._tools_registry.items()
                ]
                return ListToolsObservation(tools=tools, done=False, reward=0.0)
            if isinstance(action, CallToolAction):
                fn = self._tools_registry.get(action.tool_name)
                if fn is None:
                    return CallToolObservation(
                        tool_name=action.tool_name,
                        result=None,
                        error=ToolError(
                            error_type=ToolErrorType.TOOL_NOT_FOUND,
                            message=f"Unknown tool: {action.tool_name!r}",
                        ),
                        done=False,
                        reward=0.0,
                    )
                sig = inspect.signature(fn)
                try:
                    sig.bind(**(action.arguments or {}))
                except TypeError as exc:
                    return CallToolObservation(
                        tool_name=action.tool_name,
                        result=None,
                        error=ToolError(
                            error_type=ToolErrorType.INVALID_ARGS,
                            message=str(exc),
                        ),
                        done=False,
                        reward=0.0,
                    )
                try:
                    res = fn(**(action.arguments or {}))
                except Exception as exc:
                    return CallToolObservation(
                        tool_name=action.tool_name,
                        result=None,
                        error=ToolError(
                            error_type=ToolErrorType.EXECUTION_ERROR,
                            message=str(exc),
                        ),
                        done=False,
                        reward=0.0,
                    )
                return CallToolObservation(
                    tool_name=action.tool_name,
                    result=res,
                    error=None,
                    done=getattr(self, "_done", False),
                    reward=getattr(self, "_reward", 0.0),
                )
            return self._step_impl(action, timeout_s=timeout_s, **kwargs)

    def _module(name, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod
        return mod

    _module("openenv")
    core = _module("openenv.core", EnvClient=EnvClient)
    _module("openenv.core.client_types", StepResult=StepResult)
    env_server = _module(
        "openenv.core.env_server", create_app=MagicMock(name="create_app")
    )
    _module(
        "openenv.core.env_server.types",
        Action=Action,
        Observation=Observation,
        State=State,
    )
    _module("openenv.core.env_server.interfaces", Environment=Environment)
    _module(
        "openenv.core.env_server.mcp_types",
        ToolErrorType=ToolErrorType,
        ToolError=ToolError,
        Tool=Tool,
        ListToolsAction=ListToolsAction,
        ListToolsObservation=ListToolsObservation,
        CallToolAction=CallToolAction,
        CallToolObservation=CallToolObservation,
    )
    _module(
        "openenv.core.env_server.mcp_environment",
        MCPEnvironment=MCPEnvironment,
    )
    core.env_server = env_server


try:
    import openenv  # noqa: F401
except ImportError:
    _install_openenv_stub()

