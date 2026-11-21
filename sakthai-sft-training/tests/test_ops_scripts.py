"""Unit tests for sakthai-sft-training/scripts/ops/ tooling."""

import os
import sys
import pytest

OPS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "ops"))
if OPS_DIR not in sys.path:
    sys.path.insert(0, OPS_DIR)

from smoke_test import (
    _check_argparse_with_env,
    _check_pep723_header,
    _check_python_parses,
)
from submit_job import looks_like_debug
from weekly_ops_report import (
    _bucket_script,
    _fmt_duration,
)


def test_smoke_test_python_parses(tmp_path):
    valid_file = tmp_path / "valid.py"
    valid_file.write_text("def hello():\n    return 42\n")
    # Should complete without error
    _check_python_parses(valid_file)

    invalid_file = tmp_path / "invalid.py"
    invalid_file.write_text("def invalid syntax ((\n")
    with pytest.raises(SystemExit, match="SYNTAX ERROR"):
        _check_python_parses(invalid_file)


def test_smoke_test_pep723_header(tmp_path, capsys):
    header_file = tmp_path / "with_header.py"
    header_file.write_text(
        "# /// script\n"
        "# dependencies = [\n"
        "#   \"datasets\",\n"
        "# ]\n"
        "# ///\n"
        "import datasets\n"
    )
    _check_pep723_header(header_file)
    captured = capsys.readouterr()
    assert "declares 1 deps" in captured.out

    no_header_file = tmp_path / "no_header.py"
    no_header_file.write_text("print('no header')\n")
    _check_pep723_header(no_header_file)
    captured = capsys.readouterr()
    assert "WARN: no PEP-723 header" in captured.out


def test_submit_job_looks_like_debug():
    # Debug commands that must be caught to downgrade GPU flavor
    debug_commands = [
        "echo hello",
        "echo 'hello world'",
        "python -c 'print(1+1)'",
        "python test_job.py --model foo",
        "python hello_world.py",
        "python smoke_test.py --script train.py",
        "duckdb -c 'select hello'",
    ]
    for cmd in debug_commands:
        assert looks_like_debug(cmd) is not None, f"Expected {cmd!r} to be recognized as debug"

    # Real production jobs that must NOT be downgraded
    prod_commands = [
        "python train.py --env browsergym --model Nanthasit/sakthai-context-7b-tools",
        "python eval_bench_peft.py --model Nanthasit/sakthai-context-7b-tools",
        "python eval-cross-model.py --models Qwen/Qwen2.5-7B-Instruct",
    ]
    for cmd in prod_commands:
        assert looks_like_debug(cmd) is None, f"Expected {cmd!r} to NOT be recognized as debug"


def test_weekly_ops_report_fmt_duration():
    assert _fmt_duration(45) == "45s"
    assert _fmt_duration(125) == "2m5s"
    assert _fmt_duration(3600) == "1h0m"
    assert _fmt_duration(7325) == "2h2m"


def test_weekly_ops_report_bucket_script():
    cmd_data = ["uv", "run", "/data/train_runner.py", "--lr", "1e-4"]
    assert _bucket_script(cmd_data) == "train_runner.py"

    cmd_tmp = ["python", "/tmp/quick_eval.py"]
    assert _bucket_script(cmd_tmp) == "quick_eval.py"

    cmd_scripts = ["bash", "scripts/run_bench.sh"]
    assert _bucket_script(cmd_scripts) == "run_bench.sh"

    cmd_raw = ["python", "train.py"]
    assert _bucket_script(cmd_raw) == "python"

    assert _bucket_script([]) == "(empty)"


def test_smoke_test_check_argparse_with_env(tmp_path):
    script = tmp_path / "cli_tool.py"
    script.write_text(
        "import argparse, os\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--foo', default=os.environ.get('FOO_VAL', 'bar'))\n"
        "p.parse_args()\n"
    )
    # Testing that --help succeeds
    _check_argparse_with_env(script, {"FOO_VAL": "custom"})


