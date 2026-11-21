"""Contract and unit tests for AgentToolEnv (Tier B custom environment wrapper)."""

import inspect
import os
import sys
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_tools.wrapper import (
    TASK_KEYS,
    AgentToolEnv,
    build_dataset,
    reward_func,
)


def test_agent_tools_env_contract_and_init():
    """Verify AgentToolEnv honours 0-arg __init__ and initializes state."""
    params = list(inspect.signature(AgentToolEnv).parameters)
    assert not params, f"AgentToolEnv.__init__ must take no arguments, takes {params}"

    env = AgentToolEnv()
    assert env.reward == 0.0
    assert env.done is False
    assert env._client is None
    assert hasattr(env, "reset")


def test_agent_tools_docstrings_have_google_args():
    """All public tool methods must have Google-style Args: blocks."""
    env = AgentToolEnv()
    tools = [
        name
        for name in dir(env)
        if not name.startswith("_") and name != "reset" and callable(getattr(env, name))
    ]
    expected_tools = {"run_command", "write_file", "read_file"}
    assert expected_tools.issubset(set(tools))

    for name in tools:
        method = getattr(env, name)
        doc = inspect.getdoc(method)
        assert doc, f"tool {name} has no docstring"
        assert "Args:" in doc, f"tool {name} docstring missing Args: block"
        for param in inspect.signature(method).parameters:
            assert f"{param}:" in doc, f"tool {name} docstring missing {param}:"


def test_agent_tools_env_rejection_after_done():
    """Reject tool calls once episode is marked done."""
    env = AgentToolEnv()
    env.done = True

    with pytest.raises(ValueError, match="already ended"):
        env.run_command("ls")

    with pytest.raises(ValueError, match="already ended"):
        env.write_file("test.txt", "content")

    with pytest.raises(ValueError, match="already ended"):
        env.read_file("test.txt")


def test_agent_tools_reset_and_tools_forwarding():
    """Verify reset and tools forward results, reward, and done state from client."""
    env = AgentToolEnv()

    client = MagicMock()
    step_obs = MagicMock()
    step_obs.observation.result = "Task prompt: create answer.txt"
    step_obs.observation.reward = 0.0
    step_obs.observation.done = False
    client.reset.return_value = step_obs

    env._open = MagicMock(return_value=client)
    res = env.reset(task="answer_file")
    assert res == "Task prompt: create answer.txt"
    assert env.reward == 0.0
    assert env.done is False
    client.reset.assert_called_once_with(task_key="answer_file")

    # run_command
    cmd_res = MagicMock()
    cmd_res.observation.result = "stdout: ok"
    cmd_res.observation.reward = 0.0
    cmd_res.observation.done = False
    client.run_command.return_value = cmd_res
    out = env.run_command("echo ok")
    assert out == "stdout: ok"

    # write_file
    write_res = MagicMock()
    write_res.observation.result = "wrote 5 bytes"
    write_res.observation.reward = 1.0
    write_res.observation.done = True
    client.write_file.return_value = write_res
    out = env.write_file("answer.txt", "42\n")
    assert out == "wrote 5 bytes"
    assert env.reward == 1.0
    assert env.done is True

    # read_file when client returns StepResult
    env.done = False
    read_res = MagicMock()
    read_res.observation.result = "file content"
    read_res.observation.reward = 1.0
    read_res.observation.done = False
    client.read_file.return_value = read_res
    out = env.read_file("answer.txt")
    assert out == "file content"

    # Close releases client
    env._close()
    assert env._client is None
    client.close.assert_called_once()


def test_agent_tools_reward_func():
    """Verify reward_func extracts rewards from environment instances."""
    e1, e2 = AgentToolEnv(), AgentToolEnv()
    e1.reward = 1.0
    e2.reward = 0.0
    assert reward_func([e1, e2]) == [1.0, 0.0]


def test_agent_tools_build_dataset():
    """Verify build_dataset creates conversational dataset with task column."""
    ds = build_dataset(6)
    assert len(ds) == 6
    assert set(ds.column_names) == {"prompt", "task"}
    for i in range(len(ds)):
        assert isinstance(ds[i]["prompt"], list)
        assert ds[i]["prompt"][-1]["role"] == "user"
        assert ds[i]["task"] in TASK_KEYS
