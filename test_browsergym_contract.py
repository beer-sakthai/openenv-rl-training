import sys
import os
import unittest.mock as mock

# Delete all mocked modules that might interfere with real imports
to_delete = []
for k in sys.modules:
    if isinstance(sys.modules[k], mock.MagicMock):
        to_delete.append(k)
for k in to_delete:
    del sys.modules[k]

# Mock trl before importing train.py if trl is not installed locally
sys.modules['trl'] = mock.MagicMock()

import pytest

sys.path.append(os.path.abspath("openenv-custom-training"))
from train import _browsergym_dataset, _browsergym_reward, BROWSERGYM_SPACE_URL

def test_browsergym_dataset_column_names_and_size():
    n_episodes = 5
    dataset = _browsergym_dataset(n_episodes)
    assert len(dataset) == n_episodes
    assert "prompt" in dataset.column_names
    # Check that "prompt" values contain "web navigation agent"
    assert "web navigation agent" in dataset[0]["prompt"]

def test_browsergym_reward_extraction():
    mock_env_outputs = [{"reward": 1.0}, {"reward": 0.5}, {"reward": -0.2}]
    completions = ["click('submit')", "fill('search')", "noop()"]
    rewards = _browsergym_reward(completions, env_outputs=mock_env_outputs)
    assert rewards == [1.0, 0.5, -0.2]

def test_browsergym_space_url_const():
    assert "nanthasit-browsergym-env.hf.space" in BROWSERGYM_SPACE_URL

def test_browsergym_functions_have_docstrings():
    assert _browsergym_dataset.__doc__ is not None
    assert isinstance(_browsergym_dataset.__doc__, str)
    assert len(_browsergym_dataset.__doc__.strip()) > 0

    assert _browsergym_reward.__doc__ is not None
    assert isinstance(_browsergym_reward.__doc__, str)
    assert len(_browsergym_reward.__doc__.strip()) > 0
