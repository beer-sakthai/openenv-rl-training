"""MCP-backed OpenEnv environment and TRL `environment_factory` wrapper.

Adapted from the OpenEnv `mcp_environment.ipynb` reference notebook to show how
OpenEnv's Model Context Protocol (MCP) surface (`MCPEnvironment`,
`ListToolsAction`, `CallToolAction`, `CallToolObservation`, `ToolErrorType`)
composes with TRL's `GRPOTrainer(environment_factory=...)` contract.

Key design points (from `mcp_environment.ipynb` & `CLAUDE.md`):
  - **Dual API boundary**: Training orchestration uses `reset()`, `step()`,
    `state()`; agent actions flow through `step(CallToolAction(tool_name=...,
    arguments=...))`.
  - **Error classification**: `CallToolObservation.error` carries a structured
    `ToolError` with `ToolErrorType` (`TOOL_NOT_FOUND`, `INVALID_ARGS`,
    `EXECUTION_ERROR`, `TRANSPORT_ERROR`, `TIMEOUT`). Always check
    `obs.error is None` before reading `obs.result`.
  - **TRL `environment_factory` contract**: `MCPTaskEnv.__init__(self)` takes
    no arguments, `reset(**kwargs)` receives dataset columns, every public
    method other than `reset` has a Google-style `Args:` block for
    `transformers.get_json_schema`, and episodes are capped at `MCP_STEP_LIMIT`.

Run with:  python train.py --env mcp [--model ...]
"""

from uuid import uuid4

from datasets import Dataset
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
)
from openenv.core.env_server.types import Action, Observation, State

MCP_STEP_LIMIT = 10

_RESERVED_MCP_NAMES = frozenset({"reset", "step", "state", "close"})


class _LightweightMCPServer:
    """Minimal FastMCP-compatible registry used when `fastmcp` is not installed.

    If `fastmcp` is installed on the training box, `FastMCP` is used directly;
    otherwise this lightweight decorator registry provides the same `@mcp.tool`
    registration surface for in-process simulation and CPU contract tests.
    """

    def __init__(self, name: str):
        self.name = name
        self._tools_registry: dict = {}

    def tool(self, fn):
        if fn.__name__ in _RESERVED_MCP_NAMES:
            raise ValueError(
                f"Tool name {fn.__name__!r} is reserved by the OpenEnv control plane"
            )
        self._tools_registry[fn.__name__] = fn
        return fn


def _create_mcp_server(name: str):
    try:
        from fastmcp import FastMCP  # type: ignore[import-not-found]

        return FastMCP(name)
    except ImportError:
        return _LightweightMCPServer(name)


def _extract_tool_result(obs) -> object:
    """Unwrap `CallToolObservation.result` across FastMCP and JSON round-trips."""
    result = getattr(obs, "result", None)
    if hasattr(result, "data"):
        return result.data
    if isinstance(result, dict) and "data" in result:
        return result["data"]
    return result


# Catalog of structured verification records for the MCP lookup & verification task.
ORDER_RECORDS = {
    "ORD-101": {"customer": "Somchai", "amount_thb": 1250, "status": "paid"},
    "ORD-202": {"customer": "Nanthasit", "amount_thb": 4800, "status": "shipped"},
    "ORD-303": {"customer": "Anong", "amount_thb": 920, "status": "pending"},
    "ORD-404": {"customer": "Kittisak", "amount_thb": 3100, "status": "paid"},
}


class SakThaiMCPEnvironment(MCPEnvironment):
    """In-process OpenEnv `MCPEnvironment` exposing order-lookup and verification tools."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        mcp = _create_mcp_server("sakthai_mcp_env")
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._target_order_id = "ORD-101"
        self._done = False
        self._reward = 0.0

        @mcp.tool
        def lookup_order(order_id: str) -> dict:
            """Look up an order record by its order ID.

            Args:
                order_id: The order identifier (e.g. 'ORD-101').

            Returns:
                Dictionary containing customer, amount_thb, and status.
            """
            key = order_id.strip().upper()
            if key not in ORDER_RECORDS:
                raise KeyError(f"Order {order_id!r} not found")
            return {"order_id": key, **ORDER_RECORDS[key]}

        @mcp.tool
        def compute_vat(amount_thb: int, rate_percent: int = 7) -> dict:
            """Compute Thai VAT and total inclusive amount for an order.

            Args:
                amount_thb: Base order amount in Thai Baht.
                rate_percent: VAT percentage rate (default 7).

            Returns:
                Dictionary with vat_thb and total_thb rounded to 2 decimals.
            """
            if amount_thb < 0:
                raise ValueError("amount_thb must be non-negative")
            vat = round(amount_thb * (rate_percent / 100.0), 2)
            return {
                "amount_thb": amount_thb,
                "vat_thb": vat,
                "total_thb": round(amount_thb + vat, 2),
            }

        @mcp.tool
        def submit_invoice_total(order_id: str, total_thb: float) -> str:
            """Submit the final VAT-inclusive invoice total for the target order.

            Args:
                order_id: The order identifier being submitted.
                total_thb: The computed total in THB including 7% VAT.

            Returns:
                Confirmation string indicating whether the submission matched.
            """
            key = order_id.strip().upper()
            expected_rec = ORDER_RECORDS.get(self._target_order_id)
            if expected_rec is None:
                raise RuntimeError("Target order missing from registry")
            expected_total = round(expected_rec["amount_thb"] * 1.07, 2)
            self._done = True
            if key == self._target_order_id and abs(float(total_thb) - expected_total) < 1e-2:
                self._reward = 1.0
                return f"VERIFIED: {key} total_thb={expected_total:.2f}"
            self._reward = 0.0
            return (
                f"MISMATCH: submitted ({key}, {total_thb}), "
                f"expected ({self._target_order_id}, {expected_total:.2f})"
            )

        super().__init__(mcp)

    def reset(self, seed=None, episode_id=None, **kwargs) -> Observation:
        self._target_order_id = str(kwargs.get("order_id", "ORD-101")).strip().upper()
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._done = False
        self._reward = 0.0
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "order_id": self._target_order_id,
                "prompt": (
                    f"Look up order {self._target_order_id}, compute its 7% VAT total, "
                    f"and submit the final total with submit_invoice_total."
                ),
            },
        )

    def _step_impl(self, action: Action, timeout_s=None, **kwargs) -> Observation:
        return Observation(
            done=False,
            reward=0.0,
            metadata={"error": f"Unknown non-MCP action type: {type(action).__name__}"},
        )

    @property
    def state(self) -> State:
        return self._state


class MCPTaskEnv:
    """TRL `environment_factory` wrapper driving `SakThaiMCPEnvironment` via `CallToolAction`."""

    def __init__(self):
        self._env = SakThaiMCPEnvironment()
        self._steps = 0
        self.reward = 0.0
        self.done = False

    def reset(self, **kwargs) -> str | None:
        self._steps = 0
        self.reward = 0.0
        self.done = False
        order_id = kwargs.get("order_id", "ORD-101")
        obs = self._env.reset(order_id=order_id)
        meta = getattr(obs, "metadata", None) or {}
        return meta.get(
            "prompt",
            f"Look up order {order_id}, compute its 7% VAT total, and submit it.",
        )

    def _call_mcp(self, tool_name: str, arguments: dict) -> str:
        if self.done:
            raise ValueError("Episode already ended.")
        self._steps += 1
        step_out = self._env.step(
            CallToolAction(tool_name=tool_name, arguments=arguments)
        )
        obs = getattr(step_out, "observation", step_out)
        if isinstance(obs, CallToolObservation) and obs.error is not None:
            err_type = getattr(obs.error.error_type, "value", obs.error.error_type)
            raise ValueError(f"MCP {err_type}: {obs.error.message}")

        reward = getattr(step_out, "reward", None)
        if reward is None:
            reward = getattr(obs, "reward", 0.0)
        self.reward = float(reward or 0.0)
        self.done = bool(
            getattr(step_out, "done", None) or getattr(obs, "done", False)
        )
        data = _extract_tool_result(obs)
        text = str(data)
        if not self.done and self._steps >= MCP_STEP_LIMIT:
            self.done = True
            text = f"{text}\n\nSTEP LIMIT REACHED ({MCP_STEP_LIMIT})."
        return text

    def list_available_tools(self) -> list[str]:
        """Return the names of all MCP tools registered on the underlying environment.

        Args:
            None
        """
        obs = self._env.step(ListToolsAction())
        inner = getattr(obs, "observation", obs)
        return [t.name for t in getattr(inner, "tools", [])]

    # -- TRL discoverable tools ----------------------------------------------

    def lookup_order(self, order_id: str) -> str:
        """Look up an order record by its order ID via MCP.

        Args:
            order_id: The order identifier to query (e.g. 'ORD-101').
        """
        return self._call_mcp("lookup_order", {"order_id": order_id})

    def compute_vat(self, amount_thb: int, rate_percent: int = 7) -> str:
        """Compute Thai VAT and the VAT-inclusive total for an order amount.

        Args:
            amount_thb: Base order amount in Thai Baht.
            rate_percent: VAT percentage rate to apply (default 7).
        """
        return self._call_mcp(
            "compute_vat",
            {"amount_thb": amount_thb, "rate_percent": rate_percent},
        )

    def submit_invoice_total(self, order_id: str, total_thb: float) -> str:
        """Submit the final VAT-inclusive invoice total for the target order.

        Args:
            order_id: The order identifier being submitted.
            total_thb: The computed total in THB including 7% VAT.
        """
        return self._call_mcp(
            "submit_invoice_total",
            {"order_id": order_id, "total_thb": total_thb},
        )


# Hide helper method from TRL tool discovery while keeping it accessible for tests
MCPTaskEnv._list_available_tools = MCPTaskEnv.list_available_tools
del MCPTaskEnv.list_available_tools


def reward_func(environments, **kwargs) -> list[float]:
    """Binary, outcome-based reward forwarding `env.reward` from each episode."""
    return [float(getattr(env, "reward", 0.0) or 0.0) for env in environments]


def build_dataset(n_episodes: int = 64) -> Dataset:
    """Conversational prompt dataset with deterministic `order_id` scoring column."""
    keys = tuple(ORDER_RECORDS.keys())
    prompts = []
    order_ids = []
    for i in range(n_episodes):
        oid = keys[i % len(keys)]
        prompts.append(
            [
                {
                    "role": "user",
                    "content": (
                        f"You are an accounting agent with MCP tools. Look up order {oid} "
                        "using lookup_order(order_id), calculate its 7% VAT-inclusive total "
                        "using compute_vat(amount_thb), and finish by calling "
                        "submit_invoice_total(order_id, total_thb)."
                    ),
                }
            ]
        )
        order_ids.append(oid)
    return Dataset.from_dict({"prompt": prompts, "order_id": order_ids})


ENVIRONMENT_FACTORY = MCPTaskEnv
REWARD_FUNCS = reward_func
TRAIN_DATASET = build_dataset
DEFAULT_MODEL = "Nanthasit/sakthai-context-7b-tools"
