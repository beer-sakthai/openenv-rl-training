"""Contract and unit tests for MultiCatalogEnv and train_multi_env.py."""

import inspect
import os
import sys
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_multi_env import (
    CODING_EXPECTED,
    MultiCatalogEnv,
    REWARD_FUNCS,
    build_dataset,
    echo_reward,
    openspiel_reward,
    sudoku_reward,
)


def test_multi_catalog_env_init():
    """MultiCatalogEnv.__init__ takes no arguments and initializes cleanly."""
    params = list(inspect.signature(MultiCatalogEnv).parameters)
    assert not params, f"MultiCatalogEnv.__init__ must take no arguments, takes {params}"

    env = MultiCatalogEnv()
    assert env.active is None
    assert env.reward == 0.0
    assert env.done is False


def test_multi_catalog_env_tool_docstrings_have_google_args():
    """All public tool methods must have Google-style Args: blocks."""
    env = MultiCatalogEnv()
    tools = [
        name
        for name in dir(env)
        if not name.startswith("_") and name != "reset" and callable(getattr(env, name))
    ]
    expected_tools = {
        "echo_say",
        "sudoku_move",
        "coding_run",
        "chat_reply",
        "atari_press",
        "openspiel_move",
        "repl_execute",
        "repl_finish",
        "sumo_set_phase",
    }
    assert expected_tools.issubset(set(tools))

    for name in tools:
        method = getattr(env, name)
        doc = inspect.getdoc(method)
        assert doc, f"tool {name} has no docstring"
        assert "Args:" in doc, f"tool {name} docstring missing Args: block"
        for param in inspect.signature(method).parameters:
            assert f"{param}:" in doc, f"tool {name} docstring missing {param}:"


def test_coding_run_reward():
    """Verify coding_run evaluates CODING_EXPECTED against stdout."""
    env = MultiCatalogEnv()
    env.active = "coding"
    client = MagicMock()
    step_res = MagicMock()
    step_res.observation.exit_code = 0
    step_res.observation.stdout = f"Parsed invoices.\n{CODING_EXPECTED}"
    step_res.observation.stderr = ""
    client.step.return_value = step_res
    env._clients["coding"] = client

    out = env.coding_run("run_solution()")
    assert CODING_EXPECTED in out
    assert env.reward == 1.0
    assert env.done is True


def test_multi_catalog_env_routing_rejects_wrong_tool():
    """Calling a tool that does not belong to the active env raises ValueError."""
    env = MultiCatalogEnv()
    env.active = "echo"

    with pytest.raises(ValueError, match="sudoku_move is only valid during the sudoku task"):
        env.sudoku_move("1,1=4")

    with pytest.raises(ValueError, match="coding_run is only valid during the coding task"):
        env.coding_run("print(1)")

    with pytest.raises(ValueError, match="openspiel_move is only valid during the openspiel task"):
        env.openspiel_move(0)


def test_multi_catalog_env_rejects_calls_after_done():
    """Calling tools after episode has ended raises ValueError."""
    env = MultiCatalogEnv()
    env.active = "echo"
    env.done = True

    with pytest.raises(ValueError, match="already ended"):
        env.echo_say("hello")


def test_format_openspiel_obs_covers_full_catch_grid():
    """Verify that _format_openspiel_obs handles 64 floats and includes active indices and player."""
    env = MultiCatalogEnv()
    obs = MagicMock()
    # 50-element OpenSpiel Catch grid with ball at (0, 0) and paddle at (9, 2) [index 47]
    grid = [0.0] * 64
    grid[0] = 1.0
    grid[47] = 1.0  # Paddle row index 47
    obs.info_state = grid
    obs.legal_actions = [0, 1, 2]
    obs.game_phase = "play"
    obs.current_player_id = 0
    obs.opponent_last_action = None

    formatted = env._format_openspiel_obs(obs)
    assert "Active indices: [0, 47]" in formatted
    assert "Player: 0" in formatted
    assert "Legal actions: [0, 1, 2]" in formatted


def test_format_atari_obs():
    env = MultiCatalogEnv()
    obs = MagicMock()
    obs.screen = list(range(128))
    obs.legal_actions = [0, 1, 2, 3]

    formatted = env._format_atari_obs(obs)
    assert "Atari RAM state (128 bytes):" in formatted
    assert "Legal actions: [0, 1, 2, 3]" in formatted


def test_format_sumo_obs():
    env = MultiCatalogEnv()
    obs = MagicMock()
    obs.observation = [0.5, 1.2, 0.0] * 6
    obs.action_mask = [1, 1, 0, 0]
    obs.sim_time = 42

    formatted = env._format_sumo_obs(obs)
    assert "Traffic state (truncated):" in formatted
    assert "Action mask: [1, 1, 0, 0]" in formatted
    assert "Sim time: 42" in formatted


def test_rewards_per_task():
    """Each reward func returns env.reward for its own task and None for others."""
    echo_env = MultiCatalogEnv()
    echo_env.active = "echo"
    echo_env.reward = 1.0

    sudoku_env = MultiCatalogEnv()
    sudoku_env.active = "sudoku"
    sudoku_env.reward = 0.5

    assert echo_reward([echo_env, sudoku_env]) == [1.0, None]
    assert sudoku_reward([echo_env, sudoku_env]) == [None, 0.5]
    assert openspiel_reward([echo_env, sudoku_env]) == [None, None]
    assert len(REWARD_FUNCS) == 8


def test_build_dataset():
    """Verify build_dataset creates balanced episodes with prompt and env columns."""
    ds = build_dataset(n_per_env=4)
    assert len(ds) == 32  # 8 envs * 4
    assert set(ds.column_names) == {"prompt", "env"}

    # Ensure all 8 envs are present
    envs_found = set(ds["env"])
    assert envs_found == {"echo", "sudoku", "coding", "chat", "atari", "openspiel", "repl", "sumo"}

    first_prompt = ds[0]["prompt"]
    assert isinstance(first_prompt, list)
    assert first_prompt[-1]["role"] == "user"
