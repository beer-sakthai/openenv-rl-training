"""Contract and behavior tests for `env_mcp_task.py` (OpenEnv MCP + TRL wrapper)."""

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env_mcp_task import (  # noqa: E402
    MCP_STEP_LIMIT,
    MCPTaskEnv,
    SakThaiMCPEnvironment,
    build_dataset,
    reward_func,
)
from openenv.core.env_server.mcp_types import (  # noqa: E402
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
    ToolErrorType,
)


def test_mcp_environment_lists_registered_tools():
    env = SakThaiMCPEnvironment()
    env.reset(order_id="ORD-101")
    obs = env.step(ListToolsAction())
    assert isinstance(obs, ListToolsObservation)
    names = {t.name for t in obs.tools}
    assert names == {"lookup_order", "compute_vat", "submit_invoice_total"}


def test_mcp_environment_call_tool_and_error_types():
    env = SakThaiMCPEnvironment()
    env.reset(order_id="ORD-101")

    # 1. Valid call
    obs = env.step(
        CallToolAction(tool_name="lookup_order", arguments={"order_id": "ORD-101"})
    )
    assert isinstance(obs, CallToolObservation)
    assert obs.error is None
    assert obs.result["amount_thb"] == 1250

    # 2. Unknown tool -> TOOL_NOT_FOUND
    obs_missing = env.step(
        CallToolAction(tool_name="does_not_exist", arguments={})
    )
    assert obs_missing.error is not None
    assert obs_missing.error.error_type == ToolErrorType.TOOL_NOT_FOUND

    # 3. Invalid arguments -> INVALID_ARGS
    obs_bad_args = env.step(
        CallToolAction(tool_name="lookup_order", arguments={"wrong_param": "x"})
    )
    assert obs_bad_args.error is not None
    assert obs_bad_args.error.error_type == ToolErrorType.INVALID_ARGS

    # 4. Exception inside tool -> EXECUTION_ERROR
    obs_exec_err = env.step(
        CallToolAction(tool_name="lookup_order", arguments={"order_id": "ORD-999"})
    )
    assert obs_exec_err.error is not None
    assert obs_exec_err.error.error_type == ToolErrorType.EXECUTION_ERROR


def test_mcp_task_env_trl_contract_and_docstrings():
    """Verify TRL environment_factory requirements: 0-arg init + Google Args: blocks."""
    env = MCPTaskEnv()
    assert env.reward == 0.0
    assert env.done is False

    tools = [
        name
        for name in dir(env)
        if not name.startswith("_") and name != "reset" and callable(getattr(env, name))
    ]
    assert set(tools) == {"lookup_order", "compute_vat", "submit_invoice_total"}
    for name in tools:
        method = getattr(env, name)
        doc = inspect.getdoc(method)
        assert doc and "Args:" in doc, f"{name} missing Google-style Args: docstring"
        for param in inspect.signature(method).parameters:
            assert f"{param}:" in doc, f"{name} docstring missing {param}:"


def test_mcp_task_env_happy_path_and_reward_forwarding():
    env = MCPTaskEnv()
    first_obs = env.reset(order_id="ORD-101")
    assert "ORD-101" in first_obs

    rec_str = env.lookup_order("ORD-101")
    assert "1250" in rec_str
    assert env.done is False

    vat_str = env.compute_vat(1250, 7)
    assert "1337.5" in vat_str
    assert env.done is False

    verdict = env.submit_invoice_total("ORD-101", 1337.5)
    assert "VERIFIED" in verdict
    assert env.done is True
    assert env.reward == 1.0
    assert reward_func([env]) == [1.0]


def test_mcp_task_env_step_limit_terminates_cleanly():
    env = MCPTaskEnv()
    env.reset(order_id="ORD-202")
    for _ in range(MCP_STEP_LIMIT - 1):
        env.lookup_order("ORD-202")
        assert env.done is False
    last_out = env.lookup_order("ORD-202")
    assert "STEP LIMIT REACHED" in last_out
    assert env.done is True
    assert env.reward == 0.0
    with pytest.raises(ValueError, match="already ended"):
        env.lookup_order("ORD-202")


def test_mcp_dataset_is_conversational():
    ds = build_dataset(4)
    assert len(ds) == 4
    assert set(ds.column_names) >= {"prompt", "order_id"}
    first_prompt = ds[0]["prompt"]
    assert isinstance(first_prompt, list)
    assert first_prompt[-1]["role"] == "user"
    assert "lookup_order" in first_prompt[-1]["content"]
