import pytest
import sys
import os
import unittest.mock as mock

# Mock load_dataset before importing
sys.modules['datasets'] = mock.MagicMock()

# Set required environment variables for module initialization
os.environ["SAK_MODELS"] = "dummy/model"

# Add scripts directory to path to import eval_bench_peft
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from eval_bench_peft import norm

def test_norm_empty():
    """Test handling of empty inputs in norm function."""
    # Test empty string
    assert norm("") == ""
    assert norm("   ") == ""

    # Test empty dictionary
    assert norm({}) == {}

    # Test empty list
    assert norm([]) == []

    # Test None
    assert norm(None) is None

    # Test string representations of empty structures
    assert norm("{}") == {}
    assert norm("[]") == []
