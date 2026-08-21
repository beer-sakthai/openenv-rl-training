import pytest
import sys
import os
from unittest import mock

# Setup mock so we can import eval_bench_peft without failing on datasets
sys.modules['datasets'] = mock.MagicMock()

# Setup environ mock so we don't trigger validation asserts during import
with mock.patch.dict(os.environ, {"SAK_MODELS": "test_model"}):
    # Add the scripts directory to path to allow import
    script_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts')
    sys.path.append(script_dir)
    from eval_bench_peft import parse_calls

class TestParseCalls:
    def test_valid_json(self):
        text = "<tool_call>{\"name\": \"test\", \"arguments\": {\"arg1\": 1}}</tool_call>"
        assert parse_calls(text) == [{"name": "test", "arguments": {"arg1": 1}}]

    def test_invalid_json(self):
        # Missing closing brace
        text = "<tool_call>{\"name\": \"test\", \"arguments\": {\"arg1\": 1}</tool_call>"
        assert parse_calls(text) == []

        # Malformed JSON
        text = "<tool_call>{name: \"test\", arguments: {arg1: 1}}</tool_call>"
        assert parse_calls(text) == []

    def test_valid_json_string_arguments(self):
        text = "<tool_call>{\"name\": \"test\", \"arguments\": \"{\\\"arg1\\\": 1}\"}</tool_call>"
        assert parse_calls(text) == [{"name": "test", "arguments": {"arg1": 1}}]

    def test_invalid_json_string_arguments(self):
        text = "<tool_call>{\"name\": \"test\", \"arguments\": \"{\\\"arg1\\\": 1\"}</tool_call>"
        assert parse_calls(text) == [{"name": "test", "arguments": {"__unparsed__": "{\"arg1\": 1"}}]

    def test_invalid_arguments_type(self):
        # Number instead of object/string
        text = "<tool_call>{\"name\": \"test\", \"arguments\": 123}</tool_call>"
        assert parse_calls(text) == [{"name": "test", "arguments": {}}]

        # Boolean instead of object/string
        text = "<tool_call>{\"name\": \"test\", \"arguments\": true}</tool_call>"
        assert parse_calls(text) == [{"name": "test", "arguments": {}}]

        # Null instead of object/string
        text = "<tool_call>{\"name\": \"test\", \"arguments\": null}</tool_call>"
        assert parse_calls(text) == [{"name": "test", "arguments": {}}]

    def test_missing_name(self):
        text = "<tool_call>{\"arguments\": {\"arg1\": 1}}</tool_call>"
        assert parse_calls(text) == []

    def test_missing_arguments(self):
        text = "<tool_call>{\"name\": \"test\"}</tool_call>"
        assert parse_calls(text) == [{"name": "test", "arguments": {}}]

    def test_multiple_tool_calls(self):
        text = '''
        <tool_call>{"name": "test1", "arguments": {"arg1": 1}}</tool_call>
        Some random text between calls
        <tool_call>{"name": "test2", "arguments": {"arg2": 2}}</tool_call>
        '''
        assert parse_calls(text) == [
            {"name": "test1", "arguments": {"arg1": 1}},
            {"name": "test2", "arguments": {"arg2": 2}}
        ]

    def test_mixed_valid_invalid_tool_calls(self):
        text = '''
        <tool_call>{"name": "test1", "arguments": {"arg1": 1}}</tool_call>
        <tool_call>{"name": "test2", "arguments": {"arg2": 2}</tool_call>
        <tool_call>{"name": "test3", "arguments": {"arg3": 3}}</tool_call>
        '''
        assert parse_calls(text) == [
            {"name": "test1", "arguments": {"arg1": 1}},
            {"name": "test3", "arguments": {"arg3": 3}}
        ]
