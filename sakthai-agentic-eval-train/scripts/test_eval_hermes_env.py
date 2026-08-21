from eval_hermes_env import parse_tool_call

def test_parse_tool_call_valid():
    text = '<tool_call> {"name": "get_weather", "arguments": {"location": "London"}} </tool_call>'
    result = parse_tool_call(text)
    assert result == {"name": "get_weather", "arguments": {"location": "London"}}

def test_parse_tool_call_valid_with_newlines():
    text = '''<tool_call>
{
  "name": "get_weather",
  "arguments": {
    "location": "London"
  }
}
</tool_call>'''
    result = parse_tool_call(text)
    assert result == {"name": "get_weather", "arguments": {"location": "London"}}

def test_parse_tool_call_missing_tags():
    text = '{"name": "get_weather", "arguments": {"location": "London"}}'
    result = parse_tool_call(text)
    assert result is None

def test_parse_tool_call_invalid_json():
    text = '<tool_call> {"name": "get_weather", "arguments": {"location": "London"} </tool_call>' # missing closing brace
    result = parse_tool_call(text)
    assert result is None

def test_parse_tool_call_empty_string():
    text = ''
    result = parse_tool_call(text)
    assert result is None

def test_parse_tool_call_extra_text_around_tags():
    text = 'Some introductory text <tool_call> {"name": "get_weather"} </tool_call> Some trailing text'
    result = parse_tool_call(text)
    assert result == {"name": "get_weather"}
