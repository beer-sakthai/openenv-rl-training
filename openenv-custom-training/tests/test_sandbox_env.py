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

def _popen_mock(returncode=0, stdout="", stderr=""):
    """A mock subprocess.Popen whose communicate() returns one (stdout, stderr)."""
    mock_proc = MagicMock()
    mock_proc.returncode = returncode
    mock_proc.communicate.return_value = (stdout, stderr)
    return mock_proc

def test_run_command_empty(env):
    with pytest.raises(ValueError, match="empty command"):
        env._run_command("   ")

def test_run_command_success(env):
    with patch("subprocess.Popen", return_value=_popen_mock(0, "hello", "")):
        out, success = env._run_command("echo hello")

    assert success is True
    assert "exit_code: 0" in out
    assert "stdout:\nhello" in out

def test_run_command_non_zero_exit(env):
    with patch("subprocess.Popen", return_value=_popen_mock(1, "", "error message")):
        out, success = env._run_command("some_bad_command")

    assert success is False
    assert "exit_code: 1" in out
    assert "stderr:\nerror message" in out

def test_run_command_stdout_truncation(env):
    with patch("subprocess.Popen", return_value=_popen_mock(0, "a" * (MAX_STDOUT + 10), "")):
        out, success = env._run_command("echo long_string")

    assert success is True
    assert len(out) < MAX_STDOUT + 100
    assert "…" in out

def test_run_command_stderr_truncation(env):
    with patch("subprocess.Popen", return_value=_popen_mock(1, "", "b" * (MAX_STDERR + 10))):
        out, success = env._run_command("echo long_error")

    assert success is False
    assert len(out) < MAX_STDERR + 100
    assert "…" in out

def test_run_command_timeout(env):
    mock_proc = MagicMock()
    mock_proc.communicate.side_effect = subprocess.TimeoutExpired(cmd="sleep 100", timeout=10)
    with patch("subprocess.Popen", return_value=mock_proc), \
            patch("os.getpgid", return_value=4242), patch("os.killpg"):
        with pytest.raises(subprocess.TimeoutExpired):
            env._run_command("sleep 100")

def test_run_command_timeout_kills_the_whole_process_group(env):
    """A timed-out command must SIGKILL its process group, not just bash, so a
    background child it spawned (e.g. `sleep 999 &`) cannot survive the step."""
    import signal

    mock_proc = MagicMock()
    mock_proc.pid = 4242
    mock_proc.communicate.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=10)
    with patch("subprocess.Popen", return_value=mock_proc) as popen, \
            patch("os.getpgid", return_value=4242) as getpgid, \
            patch("os.killpg") as killpg:
        with pytest.raises(subprocess.TimeoutExpired):
            env._run_command("sleep 999 & wait")

    getpgid.assert_called_once_with(4242)
    killpg.assert_called_once_with(4242, signal.SIGKILL)
    # start_new_session=True is what puts the command in its own process group.
    assert popen.call_args.kwargs.get("start_new_session") is True

def test_reset_rejects_unknown_task_key(env):
    with pytest.raises(ValueError, match="unknown task_key"):
        env.reset(task_key="does_not_exist")

def test_confine_success(env):
    path = env._confine("inside.txt")
    assert path.name == "inside.txt"
    assert path.is_relative_to(env._workdir.resolve())

def test_confine_escape(env):
    with pytest.raises(ValueError, match="path escapes the sandbox workspace"):
        env._confine("../outside.txt")

def test_confine_escape_absolute(env):
    with pytest.raises(ValueError, match="path escapes the sandbox workspace"):
        env._confine("/etc/passwd")
