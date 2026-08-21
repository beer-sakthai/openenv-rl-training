import pytest
import sys
import os

# Create a mock for missing dependencies to allow import
from unittest.mock import MagicMock
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['datasets'] = MagicMock()
sys.modules['huggingface_hub'] = MagicMock()
sys.modules['peft'] = MagicMock()

# Set required environment variable for module load
os.environ["SAK_MODELS"] = "dummy"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from eval_bench_peft import arguments_ok, _match_all, args_match, norm

def test_norm():
    """Test the normalization function which parses strings to primitives and floats."""
    assert norm('{"a": 1}') == {'a': 1.0}
    assert norm(True) is True
    assert norm(1) == 1.0
    assert norm({"b": 2, "a": 1}) == {"a": 1.0, "b": 2.0} # dictionaries should be sorted, wait no, they are just evaluated
    assert norm([1, 2]) == [1.0, 2.0]
    # Edge cases
    assert norm("  hello  ") == "hello"
    assert norm(None) is None
    # Nested dicts
    assert norm({"a": {"b": "2"}}) == {"a": {"b": 2.0}}
    assert norm({"a": ["1", "2"]}) == {"a": [1.0, 2.0]}


def test_args_match():
    """Test matching arguments which rely on norm."""
    assert args_match({"a": 1}, {"a": 1}) is True
    assert args_match({"a": 1}, {"a": "1"}) is True # "1" gets parsed to float 1.0 by norm
    assert args_match({"a": 1}, {"a": 2}) is False
    assert args_match({"a": 1, "b": 2}, {"b": 2, "a": 1}) is True # Key order shouldn't matter for equality of dicts
    assert args_match({}, {}) is True

def test_match_all():
    """Test subset matching of gold calls in predictions."""
    gold = [{"name": "f1", "arguments": {"a": 1}}]
    pred = [{"name": "f1", "arguments": {"a": 1}}]
    assert _match_all(gold, pred) is True

    # Extra calls in prediction is okay
    pred_extra = [{"name": "f1", "arguments": {"a": 1}}, {"name": "f2", "arguments": {}}]
    assert _match_all(gold, pred_extra) is True

    # Missing calls from gold is not okay
    gold_missing = [{"name": "f1", "arguments": {"a": 1}}, {"name": "f2", "arguments": {}}]
    assert _match_all(gold_missing, pred) is False

    # Multiple of same call
    gold_multi = [{"name": "f1", "arguments": {"a": 1}}, {"name": "f1", "arguments": {"a": 1}}]
    pred_single = [{"name": "f1", "arguments": {"a": 1}}]
    assert _match_all(gold_multi, pred_single) is False

    pred_multi = [{"name": "f1", "arguments": {"a": 1}}, {"name": "f1", "arguments": {"a": 1}}]
    assert _match_all(gold_multi, pred_multi) is True

    # Same name but wrong arguments
    pred_wrong_args = [{"name": "f1", "arguments": {"a": 2}}]
    assert _match_all(gold, pred_wrong_args) is False

def test_arguments_ok():
    """Test main argument checking logic across different categories."""
    gold = [{"name": "f1", "arguments": {"a": 1}}]
    pred = [{"name": "f1", "arguments": {"a": 1}}]

    # irrelevance
    assert arguments_ok("irrelevance_tools", gold, pred) is None
    assert arguments_ok("irrelevance_no_tools", gold, pred) is None

    # simple
    assert arguments_ok("simple", gold, pred) is True
    assert arguments_ok("simple", [], pred) is False # Empty gold but simple category
    assert arguments_ok("simple", gold, [{"name": "f1", "arguments": {"a": 2}}]) is False
    # For simple, we only care that AT LEAST ONE pred matches the FIRST gold call
    pred_multiple = [{"name": "f2", "arguments": {}}, {"name": "f1", "arguments": {"a": 1}}]
    assert arguments_ok("simple", gold, pred_multiple) is True

    # parallel / other
    assert arguments_ok("parallel", gold, pred) is True
    assert arguments_ok("parallel", gold, [{"name": "f1", "arguments": {"a": 2}}]) is False

    # Test with multiple gold calls in parallel
    gold_multi = [{"name": "f1", "arguments": {"a": 1}}, {"name": "f2", "arguments": {"b": 2}}]
    pred_multi = [{"name": "f1", "arguments": {"a": 1}}, {"name": "f2", "arguments": {"b": 2}}]
    assert arguments_ok("parallel", gold_multi, pred_multi) is True

    # Missing one call
    assert arguments_ok("parallel", gold_multi, [{"name": "f1", "arguments": {"a": 1}}]) is False

    # Extra call in prediction is ignored by _match_all
    assert arguments_ok("parallel", gold, [{"name": "f1", "arguments": {"a": 1}}, {"name": "f2", "arguments": {}}]) is True
