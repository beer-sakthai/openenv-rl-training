"""Unit tests for train.py selection dispatch, boolean parsing, and observation helper."""

import argparse
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import train  # noqa: E402 — module object, needed to patch GRPOConfig/GRPOTrainer
from train import (
    _bool_env,
    _obs_text,
    _select,
)


def test_select_simple_env():
    args = argparse.Namespace(env="simple")
    factory, rewards, builder, model = _select(args)
    assert factory.__name__ == "SimpleGuessEnv"
    assert callable(rewards)
    assert callable(builder)
    assert "sakthai" in model


def test_select_agent_tools_env():
    args = argparse.Namespace(env="agent_tools")
    factory, rewards, builder, model = _select(args)
    assert factory.__name__ == "AgentToolEnv"
    assert callable(rewards)
    assert callable(builder)
    assert "sakthai" in model


def test_select_mcp_env():
    args = argparse.Namespace(env="mcp")
    factory, rewards, builder, model = _select(args)
    assert factory.__name__ == "MCPTaskEnv"
    assert callable(rewards)
    assert callable(builder)
    assert "sakthai" in model


def test_select_invalid_env_raises_system_exit():
    args = argparse.Namespace(env="unknown_env_xyz")
    with pytest.raises(SystemExit, match="unknown --env 'unknown_env_xyz'"):
        _select(args)


def test_bool_env_parsing(monkeypatch):
    assert _bool_env("NON_EXISTENT_VAR", False) is False
    assert _bool_env("NON_EXISTENT_VAR", True) is True

    for val in ("1", "true", "True", "TRUE", "yes", "YES", "on", "ON"):
        monkeypatch.setenv("TEST_VAR", val)
        assert _bool_env("TEST_VAR", False) is True

    for val in ("0", "false", "no", "off", "anything_else"):
        monkeypatch.setenv("TEST_VAR", val)
        assert _bool_env("TEST_VAR", True) is False


def test_obs_text_fallback():
    class CustomObs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    # Prefers text field
    obs1 = CustomObs(text="expected text", dom="dom content")
    assert _obs_text(obs1) == "expected text"

    # Falls back to dom if text absent
    obs2 = CustomObs(dom="dom content")
    assert _obs_text(obs2) == "dom content"

    # Falls back to str()
    obs3 = CustomObs()
    assert "CustomObs" in _obs_text(obs3)


def test_main_server_mode_wires_vllm_server_host_and_port(monkeypatch):
    """Regression for the vllm_server_url crash (CLAUDE.md: GRPOConfig's
    server-mode fields are vllm_server_host + vllm_server_port, not
    vllm_server_url). main() must pass the real fields under --vllm-mode server."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py", "--env", "simple", "--vllm-mode", "server",
            "--vllm-server-host", "gpu-box", "--vllm-server-port", "9001",
            "--n-episodes", "1",
        ],
    )
    mock_config = MagicMock()
    monkeypatch.setattr(train, "GRPOConfig", mock_config)
    monkeypatch.setattr(train, "GRPOTrainer", MagicMock())

    train.main()

    kwargs = mock_config.call_args.kwargs
    assert kwargs["vllm_server_host"] == "gpu-box"
    assert kwargs["vllm_server_port"] == 9001
    assert "vllm_server_url" not in kwargs


def test_main_colocate_mode_omits_vllm_server_kwargs(monkeypatch):
    """colocate mode (the default) must not pass vllm_server_host/port at all."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--env", "simple", "--vllm-mode", "colocate", "--n-episodes", "1"],
    )
    mock_config = MagicMock()
    monkeypatch.setattr(train, "GRPOConfig", mock_config)
    monkeypatch.setattr(train, "GRPOTrainer", MagicMock())

    train.main()

    kwargs = mock_config.call_args.kwargs
    assert "vllm_server_host" not in kwargs
    assert "vllm_server_port" not in kwargs
