"""Contract and unit tests for multi_env.py in openenv-custom-training."""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import multi_env  # noqa: E402 — module object, needed to patch GRPOConfig/GRPOTrainer
from agent_tools.wrapper import AgentToolEnv, TASK_KEYS
from env_simple_task import SimpleGuessEnv
from multi_env import (
    REWARD_FUNCS,
    agent_tools_reward,
    build_dataset,
    guess_reward,
)


def test_multi_env_reward_functions_isolate_by_type():
    """Verify reward functions return float for matching class and None for other."""
    guess_instance = SimpleGuessEnv()
    guess_instance.reward = 1.0

    agent_instance = AgentToolEnv()
    agent_instance.reward = 0.5

    batch = [guess_instance, agent_instance]

    # guess_reward scores SimpleGuessEnv only
    assert guess_reward(batch) == [1.0, None]

    # agent_tools_reward scores AgentToolEnv only
    assert agent_tools_reward(batch) == [None, 0.5]

    assert REWARD_FUNCS == [guess_reward, agent_tools_reward]


def test_multi_env_build_dataset_schema_and_values():
    """Verify build_dataset generates interleaved episodes with valid routing metadata."""
    ds = build_dataset(n_per_env=4)
    assert len(ds) == 8  # 4 guess + 4 agent_tools
    assert set(ds.column_names) == {"prompt", "environment", "target", "task"}

    envs = ds["environment"]
    targets = ds["target"]
    tasks = ds["task"]
    prompts = ds["prompt"]

    assert envs.count("guess") == 4
    assert envs.count("agent_tools") == 4

    for i in range(len(ds)):
        assert isinstance(prompts[i], list)
        assert prompts[i][-1]["role"] == "user"
        if envs[i] == "guess":
            assert isinstance(targets[i], int)
            assert 0 <= targets[i] < 100
            assert tasks[i] is None
        elif envs[i] == "agent_tools":
            assert targets[i] is None
            assert tasks[i] in TASK_KEYS


def test_main_server_mode_wires_vllm_server_host_and_port(monkeypatch):
    """Regression for the vllm_server_url crash (CLAUDE.md: GRPOConfig's
    server-mode fields are vllm_server_host + vllm_server_port, not
    vllm_server_url). main() must pass the real fields under --vllm-mode server."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multi_env.py", "--vllm-mode", "server",
            "--vllm-server-host", "gpu-box", "--vllm-server-port", "9001",
            "--n-per-env", "1",
        ],
    )
    mock_config = MagicMock()
    monkeypatch.setattr(multi_env, "GRPOConfig", mock_config)
    monkeypatch.setattr(multi_env, "GRPOTrainer", MagicMock())

    multi_env.main()

    kwargs = mock_config.call_args.kwargs
    assert kwargs["vllm_server_host"] == "gpu-box"
    assert kwargs["vllm_server_port"] == 9001
    assert "vllm_server_url" not in kwargs


def test_main_colocate_mode_omits_vllm_server_kwargs(monkeypatch):
    """colocate mode (the default) must not pass vllm_server_host/port at all."""
    monkeypatch.setattr(
        sys, "argv", ["multi_env.py", "--vllm-mode", "colocate", "--n-per-env", "1"]
    )
    mock_config = MagicMock()
    monkeypatch.setattr(multi_env, "GRPOConfig", mock_config)
    monkeypatch.setattr(multi_env, "GRPOTrainer", MagicMock())

    multi_env.main()

    kwargs = mock_config.call_args.kwargs
    assert "vllm_server_host" not in kwargs
    assert "vllm_server_port" not in kwargs
