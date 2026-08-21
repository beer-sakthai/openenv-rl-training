import os
import sys
import unittest.mock as mock

import datasets

datasets.load_dataset = mock.MagicMock()

os.environ["SAK_MODELS"] = "dummy/model"

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts"))
)

from eval_bench_peft import norm  # noqa: E402


def test_norm_empty():
    """Test handling of empty inputs in norm function."""
    assert norm("") == ""
    assert norm("   ") == ""
    assert norm({}) == {}
    assert norm([]) == []
    assert norm(None) is None
    assert norm("{}") == {}
    assert norm("[]") == []
