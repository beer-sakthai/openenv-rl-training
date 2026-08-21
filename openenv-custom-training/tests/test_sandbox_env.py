import pytest
import subprocess
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agent_tools.server.sandbox_env import SandboxEnvironment, MAX_STDOUT, MAX_STDERR

@pytest.fixture
def env():
    e = SandboxEnvironment()
    e.reset()
    return e

def test_run_command_empty(env):
    with pytest.raises(ValueError, match="empty command"):
        env._run_command("   ")

def test_run_command_success(env):
    with patch("subprocess.run") as mock_run:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "hello"
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        out, success = env._run_command("echo hello")

        assert success is True
        assert "exit_code: 0" in out
        assert "stdout:\nhello" in out

def test_run_command_non_zero_exit(env):
    with patch("subprocess.run") as mock_run:
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "error message"
        mock_run.return_value = mock_proc

        out, success = env._run_command("some_bad_command")

        assert success is False
        assert "exit_code: 1" in out
        assert "stderr:\nerror message" in out

def test_run_command_stdout_truncation(env):
    with patch("subprocess.run") as mock_run:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "a" * (MAX_STDOUT + 10)
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        out, success = env._run_command("echo long_string")

        assert success is True
        assert len(out) < MAX_STDOUT + 100
        assert "…" in out

def test_run_command_stderr_truncation(env):
    with patch("subprocess.run") as mock_run:
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "b" * (MAX_STDERR + 10)
        mock_run.return_value = mock_proc

        out, success = env._run_command("echo long_error")

        assert success is False
        assert len(out) < MAX_STDERR + 100
        assert "…" in out

def test_run_command_timeout(env):
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 100", timeout=10)

        with pytest.raises(subprocess.TimeoutExpired):
            env._run_command("sleep 100")
