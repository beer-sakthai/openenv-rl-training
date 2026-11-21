import importlib.util
import math
from pathlib import Path
import sys
from unittest.mock import MagicMock

# Stub heavy GPU/training modules during import, then clean them up so dill/datasets aren't poisoned
_mock_modules = [
    "torch",
    "transformers",
    "accelerate",
    "peft",
]
for _mod in _mock_modules:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

try:
    _script_path = Path(__file__).resolve().parent.parent / "eval-cross-model.py"
    _spec = importlib.util.spec_from_file_location("eval_cross_model", _script_path)
    ecm = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(ecm)
finally:
    for _mod in _mock_modules:
        if _mod in sys.modules and isinstance(sys.modules[_mod], MagicMock):
            del sys.modules[_mod]


def test_parse_tool_calls_standard_json():
    text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Bangkok"}}</tool_call>'
    calls = ecm.parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"
    assert calls[0]["arguments"] == {"city": "Bangkok"}


def test_parse_tool_calls_stringified_arguments():
    text = '<tool_call>{"name": "calc", "arguments": "{\\"x\\": 42}"}</tool_call>'
    calls = ecm.parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "calc"
    assert calls[0]["arguments"] == {"x": 42}


def test_parse_tool_calls_multiple():
    text = (
        "Thinking about tools...\n"
        '<tool_call>{"name": "tool_a", "arguments": {"v": 1}}</tool_call>\n'
        "and another:\n"
        '<tool_call>{"name": "tool_b", "arguments": {"v": 2}}</tool_call>'
    )
    calls = ecm.parse_tool_calls(text)
    assert len(calls) == 2
    assert calls[0]["name"] == "tool_a"
    assert calls[1]["name"] == "tool_b"


def test_parse_tool_calls_invalid_or_empty():
    assert ecm.parse_tool_calls("Just regular plain text response.") == []
    assert ecm.parse_tool_calls("<tool_call>Not valid JSON</tool_call>") == []
    assert ecm.parse_tool_calls("<tool_call></tool_call>") == []


def test_norm_args():
    # String JSON should be decoded
    assert ecm.norm_args('{"b": 2, "a": 1}') == {"a": 1, "b": 2}

    # None values should be filtered out
    assert ecm.norm_args({"x": 10, "y": None}) == {"x": 10}

    # Nested structures
    nested = {"z": {"beta": 2, "alpha": None}, "a": 1}
    assert ecm.norm_args(nested) == {"a": 1, "z": {"beta": 2}}

    # Primitive types unchanged
    assert ecm.norm_args(123) == 123
    assert ecm.norm_args("plain string") == "plain string"


def test_norm_call():
    call = {"name": "test_fn", "arguments": {"k": "v", "empty": None}}
    normed = ecm.norm_call(call)
    assert normed["name"] == "test_fn"
    assert normed["arguments"] == {"k": "v"}


def test_match_score_exact_match():
    gold = [{"name": "func_a", "arguments": {"x": 1}}]
    pred = [{"name": "func_a", "arguments": {"x": 1}}]
    correct, args_ok = ecm.match_score(gold, pred)
    assert correct is True
    assert args_ok is True


def test_match_score_order_independent():
    gold = [
        {"name": "f1", "arguments": {"a": 1}},
        {"name": "f2", "arguments": {"b": 2}},
    ]
    pred = [
        {"name": "f2", "arguments": {"b": 2}},
        {"name": "f1", "arguments": {"a": 1}},
    ]
    correct, args_ok = ecm.match_score(gold, pred)
    assert correct is True
    assert args_ok is True


def test_match_score_both_empty():
    correct, args_ok = ecm.match_score([], [])
    assert correct is True
    assert args_ok is True


def test_match_score_mismatched_arguments():
    gold = [{"name": "f1", "arguments": {"a": 1}}]
    pred = [{"name": "f1", "arguments": {"a": 2}}]
    correct, args_ok = ecm.match_score(gold, pred)
    assert correct is False
    assert args_ok is False


def test_match_score_mismatched_name():
    gold = [{"name": "f1", "arguments": {"a": 1}}]
    pred = [{"name": "f2", "arguments": {"a": 1}}]
    correct, args_ok = ecm.match_score(gold, pred)
    assert correct is False
    assert args_ok is False


def test_match_score_empty_pred_nonempty_gold():
    gold = [{"name": "f1", "arguments": {"a": 1}}]
    pred = []
    correct, args_ok = ecm.match_score(gold, pred)
    assert correct is False
    assert args_ok is False


def test_pct():
    assert ecm.pct([True, True, False, False]) == 50.0
    assert ecm.pct([1, 1, 1]) == 100.0
    assert math.isnan(ecm.pct([]))


def test_summarize():
    rows = {
        "simple": {
            "sel": [True, True, False],
            "args": [True, False, False],
            "strict": [True, False, False],
        }
    }
    summary = ecm.summarize(rows)
    assert "simple" in summary
    assert summary["simple"]["n"] == 3
    assert abs(summary["simple"]["selection"] - 66.666) < 0.1
    assert abs(summary["simple"]["arguments"] - 33.333) < 0.1
    assert abs(summary["simple"]["strict"] - 33.333) < 0.1


def test_generate_via_hf_router_insecure_scheme(monkeypatch):
    import pytest

    monkeypatch.setattr(ecm, "HF_ROUTER_URL", "ftp://evil.com")
    monkeypatch.setenv("HF_TOKEN", "mock-token")
    with pytest.raises(ValueError, match="Insecure URL scheme"):
        ecm._generate_via_hf_router("dummy-model", [{"role": "user", "content": "hi"}])
