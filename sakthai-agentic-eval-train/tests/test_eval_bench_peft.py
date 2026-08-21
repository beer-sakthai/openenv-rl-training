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
