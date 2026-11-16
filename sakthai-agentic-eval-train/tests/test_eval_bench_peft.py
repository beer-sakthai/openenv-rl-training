import os
import sys
from unittest.mock import MagicMock

# Set required environment variables to pass the assertions in eval_bench_peft.py
os.environ["SAK_MODELS"] = "dummy/model"

# Add the parent directory of 'scripts' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Patch out sys modules and the load_dataset so we don't break the environment
_mock_modules = [
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "huggingface_hub",
]
for mod in _mock_modules:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import datasets  # noqa: E402

_original_load_dataset = datasets.load_dataset
datasets.load_dataset = MagicMock()

try:
    from scripts import eval_bench_peft  # noqa: E402
    from scripts.eval_bench_peft import is_degenerate  # noqa: E402
finally:
    for mod in _mock_modules:
        if mod in sys.modules and isinstance(sys.modules[mod], MagicMock):
            del sys.modules[mod]
    datasets.load_dataset = _original_load_dataset


def test_is_degenerate_short_string():
    assert not is_degenerate("short string")


def test_is_degenerate_normal_long_string():
    assert not is_degenerate(
        "This is a normal long string that shouldn't be considered degenerate."
    )


def test_is_degenerate_repeated_chars():
    assert is_degenerate("A" * 20)


def test_is_degenerate_mostly_repeated_chars():
    assert is_degenerate("A" * 18 + "BC")


def test_is_degenerate_just_below_threshold():
    # 17 'A's out of 20 chars = 0.85 (below 0.9)
    assert not is_degenerate("A" * 17 + "BCD")


def test_is_degenerate_with_spaces():
    # "A " * 20 -> 40 chars, but spaces are removed by "".join(raw.split())
    # So it becomes "A" * 20 which is degenerate
    assert is_degenerate("A " * 20)


def test_load_rows_falls_back_to_jsonl_on_json_feature_error(monkeypatch):
    """Regression guard for the Colab `Feature type 'Json' not found` crash."""
    import io
    import json
    import urllib.request

    def _raise_json_feature_error(*args, **kwargs):
        raise ValueError("Feature type 'Json' not found. Available feature types: ['Value']")

    sample_row = {
        "messages": [
            {"role": "user", "content": "Check the weather in Bangkok."},
            {"role": "assistant", "content": "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Bangkok\"}}</tool_call>"},
        ],
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
        "category": "simple",
        "gold_calls": [{"name": "get_weather", "arguments": {"city": "Bangkok"}}],
        "held_out_tool": False,
        "multi_turn": False,
        "in_v1": True,
    }
    payload = (json.dumps(sample_row) + "\n").encode("utf-8")

    class _DummyResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(eval_bench_peft, "load_dataset", _raise_json_feature_error)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda url, timeout=30: _DummyResponse(payload),
    )

    eval_bench_peft.load_rows()
    assert len(eval_bench_peft.ROWS) == 1
    assert eval_bench_peft.ROWS[0]["cat"] == "simple"
    assert eval_bench_peft.ROWS[0]["gold_calls"][0]["name"] == "get_weather"

