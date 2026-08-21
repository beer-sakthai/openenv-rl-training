import pytest
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from agent_tools.server.sandbox_env import SandboxEnvironment

def test_confine_valid_path():
    env = SandboxEnvironment()
    env.reset()

    # Valid relative paths
    p1 = env._confine("test.txt")
    assert p1.is_relative_to(env._workdir.resolve())
    assert p1.name == "test.txt"

    p2 = env._confine("subdir/test.txt")
    assert p2.is_relative_to(env._workdir.resolve())
    assert p2.parent.name == "subdir"

    # Valid absolute path inside workdir
    p3 = env._confine(str(env._workdir / "absolute.txt"))
    assert p3.is_relative_to(env._workdir.resolve())
    assert p3.name == "absolute.txt"

def test_confine_invalid_path():
    env = SandboxEnvironment()
    env.reset()

    # Try to escape with ..
    with pytest.raises(ValueError, match="escapes the sandbox workspace"):
        env._confine("../test.txt")

    with pytest.raises(ValueError, match="escapes the sandbox workspace"):
        env._confine("../../etc/passwd")

    # Try to use absolute path outside workdir
    with pytest.raises(ValueError, match="escapes the sandbox workspace"):
        env._confine("/etc/passwd")

def test_confine_symlink_escape(tmp_path):
    env = SandboxEnvironment()
    env.reset()

    # Create a symlink pointing outside
    symlink_path = env._workdir / "escape_link"
    symlink_path.symlink_to("/etc")

    # Try to traverse through symlink
    with pytest.raises(ValueError, match="escapes the sandbox workspace"):
        env._confine("escape_link/passwd")
