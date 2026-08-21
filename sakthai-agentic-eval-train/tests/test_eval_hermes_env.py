import json
import os
import sys
from unittest import mock

# Add the scripts directory to sys.path to import the module
scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.append(scripts_dir)

# Mock os.environ to avoid the AssertionError from SAK_MODELS
with mock.patch.dict(os.environ, {"SAK_MODELS": "dummy"}):
    from eval_hermes_env import _tools_block


def test_tools_block_empty():
    assert _tools_block(None) == ""
    assert _tools_block([]) == ""


def test_tools_block_single_tool():
    tools = [{"name": "get_weather", "description": "Get current weather"}]
    result = _tools_block(tools)

    assert "\n\n# Tools\n\nYou may call one or more functions." in result
    assert "<tools>\n" + json.dumps(tools[0]) + "\n</tools>" in result
    assert '<tool_call>\n{"name": <name>, "arguments": <json>}\n</tool_call>' in result


def test_tools_block_multiple_tools():
    tools = [
        {"name": "get_weather", "description": "Get current weather"},
        {"name": "get_time", "description": "Get current time"},
    ]
    result = _tools_block(tools)

    expected_sigs = "\n".join(json.dumps(t) for t in tools)
    assert f"<tools>\n{expected_sigs}\n</tools>" in result


def test_tools_block_unicode_chars():
    tools = [{"name": "ทดสอบ", "description": "นี่คือการทดสอบ"}]
    result = _tools_block(tools)

    expected_sig = json.dumps(tools[0], ensure_ascii=False)
    assert f"<tools>\n{expected_sig}\n</tools>" in result
    assert "\\u" not in result  # Ensure Unicode isn't escaped
