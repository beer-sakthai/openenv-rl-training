"""CPU contract checks for `train.py --env browsergym`.

These assert the conventions CLAUDE.md declares for every environment in this
repo, because getting them wrong fails *silently* at training time rather than
loudly at import time:

  - `prompt` must be conversational. TRL's tool-calling GRPO does
    `prompt[-1]["content"]`; a bare string raises `TypeError: string indices
    must be integers`.
  - a reward function takes `(environments, **kwargs)` and reads `env.reward`.
    A reward that always returns 0.0 gives every rollout in a group the same
    score -> zero advantage -> zero gradient, which reads exactly like the
    model-capability wall in FINDINGS.md and wastes a GPU run diagnosing it.
  - every public method other than `reset` becomes a tool, and its schema is
    generated from the type hints plus a Google-style `Args:` block by
    `transformers.get_json_schema`, which raises `DocstringParsingException`
    when a parameter is missing from `Args:`.

Earlier revisions of these tests asserted the *implementation* (a bare-string
prompt, an `env_outputs=` kwarg TRL never passes) and so stayed green while
both bugs were live. Assert the contract, not the code.

Run: uv run --with datasets --with pytest pytest test_browsergym_contract.py
"""

import os
import sys
import unittest.mock as mock

import pytest

# Mock trl before importing train.py if trl is not installed locally.
sys.modules["trl"] = mock.MagicMock()

sys.path.append(os.path.abspath("openenv-custom-training"))
from train import (  # noqa: E402
    BROWSERGYM_CODER_MODEL,
    BROWSERGYM_DEFAULT_MODEL,
    BROWSERGYM_SPACE_URL,
    BROWSERGYM_STEP_LIMIT,
    _browsergym_dataset,
    _browsergym_reward,
    _make_browsergym_factory,
    _select,
)


class _FakeEnv:
    """Stand-in for a post-episode environment instance: reward lives on self."""

    def __init__(self, reward):
        self.reward = reward


# -- dataset contract --------------------------------------------------------


def test_browsergym_dataset_size_and_columns():
    dataset = _browsergym_dataset(5)
    assert len(dataset) == 5
    assert "prompt" in dataset.column_names


def test_browsergym_prompt_is_conversational_not_a_bare_string():
    """TRL does prompt[-1]["content"]; a plain string raises TypeError."""
    prompt = _browsergym_dataset(1)[0]["prompt"]
    assert isinstance(prompt, list), f"prompt must be a list of messages, got {type(prompt)}"
    assert prompt, "prompt must not be empty"
    for message in prompt:
        assert set(message) >= {"role", "content"}, message
    # The exact access TRL performs.
    assert "web navigation agent" in prompt[-1]["content"]


# -- reward contract ---------------------------------------------------------


def test_browsergym_reward_reads_reward_off_the_environments():
    environments = [_FakeEnv(1.0), _FakeEnv(0.5), _FakeEnv(-0.2)]
    assert _browsergym_reward(environments) == [1.0, 0.5, -0.2]


def test_browsergym_reward_takes_environments_not_completions():
    """Guards the regression: a (completions, env_outputs=...) signature made
    this return all-zeros for every episode, starving GRPO of signal."""
    import inspect

    first_param = list(inspect.signature(_browsergym_reward).parameters)[0]
    assert first_param == "environments", (
        f"reward funcs take `environments` first, got `{first_param}`"
    )


def test_browsergym_reward_survives_a_missing_reward_attribute():
    assert _browsergym_reward([object()]) == [0.0]


def test_browsergym_reward_is_not_uniformly_zero_for_solved_episodes():
    """A reward that cannot distinguish success from failure yields zero
    advantage across the group and therefore zero gradient."""
    rewards = _browsergym_reward([_FakeEnv(1.0), _FakeEnv(0.0)])
    assert len(set(rewards)) > 1, "reward must vary with episode outcome"


# -- environment_factory contract --------------------------------------------


def test_factory_is_callable_with_no_arguments():
    """TRL instantiates one env per generation slot: EnvironmentFactory()."""
    env = _make_browsergym_factory("click-test", BROWSERGYM_SPACE_URL)()
    assert hasattr(env, "reset")
    assert env.reward == 0.0
    assert env.done is False


def test_tool_methods_have_google_style_args_blocks():
    """get_json_schema raises DocstringParsingException without one."""
    import inspect

    env = _make_browsergym_factory("click-test", BROWSERGYM_SPACE_URL)()
    tools = [
        name
        for name in dir(env)
        if not name.startswith("_") and name != "reset" and callable(getattr(env, name))
    ]
    assert tools, "the environment exposes no tools"
    for name in tools:
        method = getattr(env, name)
        doc = inspect.getdoc(method)
        assert doc, f"{name} has no docstring"
        assert "Args:" in doc, f"{name} docstring has no Args: block"
        for param in list(inspect.signature(method).parameters):
            assert f"{param}:" in doc, f"{name} docstring omits Args entry for {param!r}"


def test_episode_is_capped():
    """Every environment caps its episode so a non-converging rollout ends."""
    assert isinstance(BROWSERGYM_STEP_LIMIT, int) and BROWSERGYM_STEP_LIMIT > 0


def test_run_action_rejects_calls_after_the_episode_ended():
    env = _make_browsergym_factory("click-test", BROWSERGYM_SPACE_URL)()
    env.done = True
    with pytest.raises(ValueError, match="already ended"):
        env.run_action("click('a12')")


# -- constants ---------------------------------------------------------------


def test_browsergym_space_url_const():
    """The default must not point at the dead Nanthasit Space.

    `Nanthasit/browsergym-env` was removed; both `/` and `/health` on
    nanthasit-browsergym-env.hf.space return 404. The default is the upstream
    catalog Space, whose OpenEnv surface was verified on 2026-09-16.
    """
    assert "openenv-browsergym-env.hf.space" in BROWSERGYM_SPACE_URL
    assert "nanthasit" not in BROWSERGYM_SPACE_URL.lower()


def test_browsergym_models_and_selection():
    """Verify default and coder models in browsergym environment selection."""
    import argparse

    assert BROWSERGYM_DEFAULT_MODEL == "Nanthasit/sakthai-context-7b-tools"
    assert BROWSERGYM_CODER_MODEL == "Nanthasit/sakthai-coder-browser"

    # Default selection yields 7B
    args_default = argparse.Namespace(env="browsergym", browsergym_model="default")
    _, _, _, model = _select(args_default)
    assert model == BROWSERGYM_DEFAULT_MODEL

    # Coder selection yields 1.5B sakthai-coder-browser
    args_coder = argparse.Namespace(env="browsergym", browsergym_model="coder")
    _, _, _, coder_model = _select(args_coder)
    assert coder_model == BROWSERGYM_CODER_MODEL


# -- client lifecycle (regression: sync wrapper + no session leak) ------------


def _install_fake_browsergym(monkeypatch):
    """Inject a fake `browsergym_env` module and return the list of opened
    (raw_client, synced_client) pairs so a test can assert on the lifecycle."""
    import types

    opened = []

    def _synced():
        s = mock.MagicMock()
        s.__enter__.return_value = s
        obs = mock.MagicMock()
        obs.observation.text = "the page"
        s.reset.return_value = obs
        return s

    def BrowserGymEnv(**kwargs):
        raw = mock.MagicMock()
        synced = _synced()
        raw.sync.return_value = synced
        opened.append((raw, synced))
        return raw

    module = types.ModuleType("browsergym_env")
    module.BrowserGymEnv = BrowserGymEnv
    module.BrowserGymAction = lambda action: {"action": action}
    monkeypatch.setitem(sys.modules, "browsergym_env", module)
    return opened


def test_browsergym_open_goes_through_the_sync_wrapper(monkeypatch):
    """`.reset()`/`.step()` live on the `.sync()` wrapper, not the raw client;
    calling them on the raw client returns coroutines and every reward is 0.0."""
    opened = _install_fake_browsergym(monkeypatch)
    env = _make_browsergym_factory("click-test", BROWSERGYM_SPACE_URL)()

    obs = env.reset()

    assert obs == "the page"
    raw, synced = opened[-1]
    raw.sync.assert_called_once()
    synced.__enter__.assert_called_once()
    synced.reset.assert_called_once()


def test_browsergym_reset_closes_the_prior_session(monkeypatch):
    """A long GRPO run resets once per episode; each reset must close the prior
    client or server sessions leak until the concurrency limit is hit."""
    opened = _install_fake_browsergym(monkeypatch)
    env = _make_browsergym_factory("click-test", BROWSERGYM_SPACE_URL)()

    env.reset()
    _, first_synced = opened[-1]
    env.reset()

    first_synced.close.assert_called_once()
    assert len(opened) == 2
