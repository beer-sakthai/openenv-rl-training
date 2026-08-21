from agent_tools.client import AgentToolClient


def test_client_parse_result():
    client = AgentToolClient(base_url="http://localhost")
    payload = {
        "observation": {
            "result": "File written successfully.",
            "success": True,
            "done": True,
            "reward": 1.0,
            "metadata": {"some_key": "some_value"},
        },
        "done": True,
        "reward": 1.0,
        "metadata": {"some_key": "some_value"},
    }

    result = client._parse_result(payload)

    assert result.done is True
    assert result.reward == 1.0
    assert result.metadata == {"some_key": "some_value"}
    assert result.observation.result == "File written successfully."
    assert result.observation.success is True
    assert result.observation.done is True
    assert result.observation.reward == 1.0
    assert result.observation.metadata == {"some_key": "some_value"}


def test_client_parse_result_fallback_keys():
    # Test cases where keys might be only in payload or only in observation
    client = AgentToolClient(base_url="http://localhost")

    payload = {
        "observation": {"result": "Output", "success": False},
        "done": True,
        "reward": 0.5,
        "metadata": {"global": "meta"},
    }

    result = client._parse_result(payload)

    assert result.observation.result == "Output"
    assert result.observation.success is False
    assert result.observation.done is True  # fallback to payload
    assert result.observation.reward == 0.5  # fallback to payload
    assert result.observation.metadata == {"global": "meta"}


def test_client_parse_result_empty():
    client = AgentToolClient(base_url="http://localhost")
    result = client._parse_result({})

    assert result.observation.result == ""
    assert result.observation.success is False
    assert result.observation.done is False
    assert result.observation.reward == 0.0
    assert result.observation.metadata == {}
