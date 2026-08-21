import json
import os
import sys
from unittest import mock

# Add the scripts directory to sys.path to import the module
scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.append(scripts_dir)

# Heavy deps are GPU-box imports; mock them so the suite runs in a CPU checkout.
for _mod in ("torch", "transformers", "huggingface_hub"):
    sys.modules[_mod] = mock.MagicMock()

# hermes-tool-use-rl-env lives in an external workspace; only _tools_block is tested.
for _mod in ("hermes_tool_env", "models", "tasks"):
    sys.modules[_mod] = mock.MagicMock()

# Mock os.environ to avoid the AssertionError from SAK_MODELS
with mock.patch.dict(os.environ, {"SAK_MODELS": "dummy"}):
    from eval_hermes_env import _tools_block, parse_tool_call


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


def test_parse_tool_call_valid():
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Bangkok"}}\n</tool_call>'
    expected = {"name": "get_weather", "arguments": {"location": "Bangkok"}}
    assert parse_tool_call(text) == expected


def test_parse_tool_call_with_surrounding_text():
    text = 'Here is the weather tool call:\n<tool_call>\n{"name": "get_weather", "arguments": {"location": "Bangkok"}}\n</tool_call>\nI hope this helps.'
    expected = {"name": "get_weather", "arguments": {"location": "Bangkok"}}
    assert parse_tool_call(text) == expected


def test_parse_tool_call_no_match():
    text = 'There is no tool call here.'
    assert parse_tool_call(text) is None


def test_parse_tool_call_invalid_json():
    text = '<tool_call>\n{"name": "get_weather", "arguments": {location": "Bangkok"}\n</tool_call>'
    assert parse_tool_call(text) is None


def test_parse_tool_call_missing_tags():
    text = '{"name": "get_weather", "arguments": {"location": "Bangkok"}}'
    assert parse_tool_call(text) is None
