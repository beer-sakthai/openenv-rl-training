import sys
import os
import unittest.mock as mock
import pytest

# Mock trl before importing train.py if trl is not installed locally
sys.modules['trl'] = mock.MagicMock()

sys.path.append(os.path.abspath("openenv-custom-training"))
from train import _browsergym_dataset, _browsergym_reward, BROWSERGYM_SPACE_URL

def test_browsergym_dataset_column_names_and_size():
    n_episodes = 5
    dataset = _browsergym_dataset(n_episodes)
    assert len(dataset) == n_episodes
    assert "prompt" in dataset.column_names
    assert "web navigation agent" in dataset[0]["prompt"]

def test_browsergym_reward_extraction():
    mock_env_outputs = [{"reward": 1.0}, {"reward": 0.5}, {"reward": -0.2}]
    completions = ["click('submit')", "fill('search')", "noop()"]
    rewards = _browsergym_reward(completions, env_outputs=mock_env_outputs)
    assert rewards == [1.0, 0.5, -0.2]

def test_browsergym_space_url_const():
    assert "nanthasit-browsergym-env.hf.space" in BROWSERGYM_SPACE_URL
