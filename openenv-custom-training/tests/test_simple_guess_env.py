"""Contract and behavior tests for SimpleGuessEnv (Tier A custom environment)."""

import inspect
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env_simple_task import (
    MAX_ATTEMPTS,
    SimpleGuessEnv,
    build_dataset,
    reward_func,
)


def test_simple_guess_env_contract():
    """Verify SimpleGuessEnv honours 0-arg __init__ and initializes state."""
    params = list(inspect.signature(SimpleGuessEnv).parameters)
    assert not params, f"SimpleGuessEnv.__init__ must take no arguments, takes {params}"

    env = SimpleGuessEnv()
    assert env.reward == 0.0
    assert env.done is False
    assert hasattr(env, "reset")


def test_simple_guess_tools_docstrings():
    """Every public method other than reset must have Google-style Args: block."""
    env = SimpleGuessEnv()
    tools = [
        name
        for name in dir(env)
        if not name.startswith("_") and name != "reset" and callable(getattr(env, name))
    ]
    assert "guess" in tools
    assert "log_reasoning" in tools

    for name in tools:
        method = getattr(env, name)
        doc = inspect.getdoc(method)
        assert doc, f"tool {name} has no docstring"
        assert "Args:" in doc, f"tool {name} docstring missing Args: block"
        for param in inspect.signature(method).parameters:
            assert f"{param}:" in doc, f"tool {name} docstring missing {param}:"


def test_simple_guess_env_rejection_after_done():
    """Reject calls to guess and log_reasoning after episode completion."""
    env = SimpleGuessEnv()
    env.reset(target=50)
    env.done = True

    with pytest.raises(ValueError, match="already ended"):
        env.guess(50)

    with pytest.raises(ValueError, match="already ended"):
        env.log_reasoning("my reasoning")


def test_simple_guess_env_out_of_bounds():
    """Reject out-of-range guesses with ValueError."""
    env = SimpleGuessEnv()
    env.reset(target=50)

    with pytest.raises(ValueError, match="out of range"):
        env.guess(-5)

    with pytest.raises(ValueError, match="out of range"):
        env.guess(105)


def test_simple_guess_env_attempts_limit():
    """Unsolved episode terminates at MAX_ATTEMPTS with reward 0.0."""
    env = SimpleGuessEnv()
    env.reset(target=42)

    for _ in range(MAX_ATTEMPTS):
        if env.done:
            break
        env.guess(10)

    assert env.done is True
    assert env.reward == 0.0


def test_simple_guess_env_bisection_solve():
    """Bisecting episode finds target and sets reward 1.0."""
    target = 73
    env = SimpleGuessEnv()
    obs = env.reset(target=target)
    assert "between 0 and 99" in obs

    low, high = 0, 99
    while not env.done:
        mid = (low + high) // 2
        feedback = env.guess(mid)
        if feedback == "higher":
            low = mid + 1
        elif feedback == "lower":
            high = mid - 1

    assert env.done is True
    assert env.reward == 1.0


def test_simple_guess_reward_func():
    """Reward function reads env.reward off each environment."""
    e1, e2 = SimpleGuessEnv(), SimpleGuessEnv()
    e1.reward = 1.0
    e2.reward = 0.0
    assert reward_func([e1, e2]) == [1.0, 0.0]


def test_simple_guess_build_dataset():
    """Dataset has conversational prompts and scoring target column."""
    ds = build_dataset(8)
    assert len(ds) == 8
    assert "prompt" in ds.column_names
    assert "target" in ds.column_names

    first_prompt = ds[0]["prompt"]
    assert isinstance(first_prompt, list)
    assert first_prompt[-1]["role"] == "user"
    assert "guess(number)" in first_prompt[-1]["content"]
