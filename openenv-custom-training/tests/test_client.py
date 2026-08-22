import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import agent_tools
sys.path.append(str(Path(__file__).parent.parent))

from agent_tools.client import AgentToolClient
from agent_tools.models import AgentToolAction, AgentToolObservation
from openenv.core.client_types import StepResult

@pytest.fixture
def mock_client():
    # Use a dummy base URL and bypass the actual EnvClient initializations that might require network
    # We can just create it without network by patching httpx or passing dummy url
    client = AgentToolClient(base_url="http://dummy")
    # Mock the step method directly on the instance
    client.step = MagicMock()

    # Setup dummy return value for step
    dummy_obs = AgentToolObservation(result="ok", success=True, done=False, reward=0.0)
    dummy_result = StepResult(observation=dummy_obs, reward=0.0, done=False, metadata={})
    client.step.return_value = dummy_result

    return client

def test_run_command_helper(mock_client):
    cmd = "echo hello"
    result = mock_client.run_command(cmd)

    # Ensure step was called once
    mock_client.step.assert_called_once()

    # Get the action passed to step
    action = mock_client.step.call_args[0][0]

    assert isinstance(action, AgentToolAction)
    assert action.tool == "run_command"
    assert action.args == {"command": cmd}

    # Ensure it returns whatever step returns
    assert result == mock_client.step.return_value

def test_write_file_helper(mock_client):
    path = "/tmp/test.txt"
    content = "hello world"
    result = mock_client.write_file(path, content)

    mock_client.step.assert_called_once()
    action = mock_client.step.call_args[0][0]

    assert isinstance(action, AgentToolAction)
    assert action.tool == "write_file"
    assert action.args == {"path": path, "content": content}
    assert result == mock_client.step.return_value

def test_read_file_helper(mock_client):
    path = "/tmp/test.txt"
    result = mock_client.read_file(path)

    mock_client.step.assert_called_once()
    action = mock_client.step.call_args[0][0]

    assert isinstance(action, AgentToolAction)
    assert action.tool == "read_file"
    assert action.args == {"path": path}
    assert result == mock_client.step.return_value
