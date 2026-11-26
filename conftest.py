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


def _install_a2a_and_catalog_stubs():
    """Register lightweight stubs for `a2a` and catalog environments if absent."""
    if "a2a" not in sys.modules:
        try:
            import a2a  # noqa: F401
        except ImportError:
            class AgentExecutor:
                pass

            class RequestContext:
                def __init__(self, task_id="task_1", context_id="ctx_1", current_task=None, message=None):
                    self.task_id = task_id
                    self.context_id = context_id
                    self.current_task = current_task
                    self.message = message

                def get_user_input(self):
                    if self.message and hasattr(self.message, "parts") and self.message.parts:
                        p = self.message.parts[0]
                        return getattr(p, "text", str(p))
                    return ""

            class EventQueue:
                def __init__(self):
                    self.events = []

                async def put(self, event):
                    self.events.append(event)

            class TaskUpdater:
                def __init__(self, event_queue=None, task_id="task_1", context_id="ctx_1"):
                    self.event_queue = event_queue
                    self.task_id = task_id
                    self.context_id = context_id
                    self.state = None
                    self.messages = []
                    self.calls = []

                async def submit(self):
                    self.calls.append("submit")

                async def start_work(self):
                    self.calls.append("start_work")

                async def update_status(self, state: str, message=None):
                    self.calls.append(("update_status", state, message))
                    self.state = state
                    if message:
                        self.messages.append(message)

                async def failed(self, message=None):
                    self.calls.append(("failed", message))
                    self.state = "failed"
                    if message:
                        self.messages.append(message)

                async def cancel(self):
                    self.calls.append("cancel")
                    self.state = "canceled"

            class DataPart:
                def __init__(self, data=None):
                    self.data = data

            class TextPart:
                def __init__(self, text=""):
                    self.text = text

            class Message:
                def __init__(self, parts=None, metadata=None, role="user", message_id=None, context_id=None, task_id=None, **kwargs):
                    self.parts = parts or []
                    self.metadata = metadata or {}
                    self.role = role
                    self.message_id = message_id
                    self.context_id = context_id
                    self.task_id = task_id
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            def new_agent_text_message(text: str, context_id=None, task_id=None):
                return Message(parts=[TextPart(text)], metadata={"context_id": context_id, "task_id": task_id})

            class Part:
                def __init__(self, text="", data=None):
                    self.text = text
                    self.data = data

            class SendMessageRequest:
                def __init__(self, message=None):
                    self.message = message

            class AgentCard:
                def __init__(self, name="OpenEnv Multi-Catalog Agent", skills=None, **kwargs):
                    self.name = name
                    self.skills = skills or ["echo", "sudoku", "coding", "chat", "atari", "openspiel", "repl", "sumo"]
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            class AgentCapabilities:
                def __init__(self, streaming=False, **kwargs):
                    self.streaming = streaming
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            class AgentSkill:
                def __init__(self, id=None, name=None, description=None, tags=None, examples=None, **kwargs):
                    self.id = id
                    self.name = name
                    self.description = description
                    self.tags = tags or []
                    self.examples = examples or []
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            class A2AStarletteApplication:
                def __init__(self, agent_card=None, http_handler=None, **kwargs):
                    self.agent_card = agent_card
                    self.http_handler = http_handler

                def build(self):
                    return self

            class DefaultRequestHandler:
                def __init__(self, agent_executor=None, task_store=None, **kwargs):
                    self.agent_executor = agent_executor
                    self.task_store = task_store

            class InMemoryTaskStore:
                def __init__(self, *args, **kwargs):
                    pass

            class A2ACardResolver:
                def __init__(self, http_client=None, base_url=""):
                    self.http_client = http_client
                    self.base_url = base_url

                async def get_agent_card(self):
                    return AgentCard()

            class TaskResult:
                def __init__(self, task_id="task_1", context_id="ctx_1"):
                    self.id = task_id
                    self.context_id = context_id

            class RootResult:
                def __init__(self, result=None):
                    self.result = result or TaskResult()

            class SendMessageResponse:
                def __init__(self, root=None):
                    self.root = root or RootResult()

            class Client:
                def __init__(self, http_client=None, agent_card=None):
                    self.http_client = http_client
                    self.agent_card = agent_card

                async def send_message(self, request):
                    return SendMessageResponse()

            def _module(name, **attrs):
                mod = types.ModuleType(name)
                for key, value in attrs.items():
                    setattr(mod, key, value)
                sys.modules[name] = mod
                return mod

            _module("a2a")
            _module(
                "a2a.types",
                DataPart=DataPart, TextPart=TextPart, Part=Part, Message=Message,
                SendMessageRequest=SendMessageRequest, AgentCard=AgentCard,
                AgentCapabilities=AgentCapabilities, AgentSkill=AgentSkill,
            )
            _module("a2a.client", A2ACardResolver=A2ACardResolver, Client=Client)
            _module("a2a.utils", new_agent_text_message=new_agent_text_message)
            _module("a2a.server")
            _module("a2a.server.apps", A2AStarletteApplication=A2AStarletteApplication)
            _module("a2a.server.events", EventQueue=EventQueue)
            _module("a2a.server.tasks", TaskUpdater=TaskUpdater, InMemoryTaskStore=InMemoryTaskStore)
            _module("a2a.server.request_handlers", DefaultRequestHandler=DefaultRequestHandler)
            _module("a2a.server.agent_execution", AgentExecutor=AgentExecutor, RequestContext=RequestContext)

    catalog_modules = [
        ("echo_env", ["EchoEnv", "EchoAction"]),
        ("textarena_env", ["TextArenaEnv", "TextArenaAction"]),
        ("coding_env", ["CodingEnv", "CodeAction"]),
        ("atari_env", ["AtariEnv", "AtariAction"]),
        ("openspiel_env", ["OpenSpielEnv", "OpenSpielAction"]),
        ("repl_env", ["REPLEnv", "REPLAction"]),
    ]
    for mod_name, classes in catalog_modules:
        if mod_name not in sys.modules:
            try:
                __import__(mod_name)
            except ImportError:
                mod = types.ModuleType(mod_name)
                for cls_name in classes:
                    setattr(mod, cls_name, MagicMock(name=cls_name))
                sys.modules[mod_name] = mod

    if "envs.chat_env" not in sys.modules:
        try:
            __import__("envs.chat_env")
        except ImportError:
            envs_mod = sys.modules.get("envs") or types.ModuleType("envs")
            sys.modules["envs"] = envs_mod
            chat_mod = types.ModuleType("envs.chat_env")
            chat_mod.ChatEnv = MagicMock(name="ChatEnv")
            chat_mod.ChatAction = MagicMock(name="ChatAction")
            sys.modules["envs.chat_env"] = chat_mod
            envs_mod.chat_env = chat_mod

    if "envs.sumo_rl_env" not in sys.modules:
        try:
            __import__("envs.sumo_rl_env")
        except ImportError:
            envs_mod = sys.modules.get("envs") or types.ModuleType("envs")
            sys.modules["envs"] = envs_mod
            sumo_mod = types.ModuleType("envs.sumo_rl_env")
            sumo_mod.SumoRLEnv = MagicMock(name="SumoRLEnv")
            sumo_mod.SumoAction = MagicMock(name="SumoAction")
            sys.modules["envs.sumo_rl_env"] = sumo_mod
            envs_mod.sumo_rl_env = sumo_mod


def _install_trl_stub():
    """Register lightweight stubs for `trl` if not installed."""
    if "trl" not in sys.modules:
        try:
            import trl  # noqa: F401
        except ImportError:
            class GRPOConfig:
                def __init__(self, *args, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            class GRPOTrainer:
                def __init__(self, *args, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            mod = types.ModuleType("trl")
            mod.GRPOConfig = GRPOConfig
            mod.GRPOTrainer = GRPOTrainer
            sys.modules["trl"] = mod


try:
    import openenv  # noqa: F401
except ImportError:
    _install_openenv_stub()

_install_a2a_and_catalog_stubs()
_install_trl_stub()

