import ast
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from pipeline_common import (  # noqa: E402
    RENDER_MODES,
    TOOLS,
    build_manifest,
    parse_tool_call,
    parse_tool_calls,
    positive_int,
    render_prompt,
    validate_choice,
)


def test_shared_tool_contract_is_rendered_with_unicode_and_nested_arguments():
    messages = [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "แก้ไฟล์"},
    ]
    prompt = render_prompt(messages, TOOLS)
    assert "<tools>" in prompt
    assert "terminal" in prompt
    assert "แก้ไฟล์" in prompt


def test_parser_accepts_multiple_calls_and_nested_json():
    raw = (
        '<tool_call>{"name":"patch","arguments":{"path":"a.py",'
        '"new_string":{"nested":true}}}</tool_call>'
        '<tool_call>{"name":"submit","arguments":{}}</tool_call>'
    )
    calls = parse_tool_calls(raw)
    assert calls[0]["arguments"]["new_string"] == {"nested": True}
    assert parse_tool_call(raw) == calls[0]


def test_validation_rejects_invalid_runtime_settings():
    assert validate_choice("renderer", "NATIVE", RENDER_MODES) == "native"
    assert positive_int("steps", "12") == 12
    for callback in (
        lambda: validate_choice("renderer", "other", RENDER_MODES),
        lambda: positive_int("steps", 0),
        lambda: positive_int("steps", "not-an-int"),
    ):
        try:
            callback()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid setting was accepted")


def test_manifest_contains_reproducibility_fields():
    manifest = build_manifest(
        model="org/model",
        benchmark="org/benchmark",
        renderer="native",
        max_steps=12,
        max_completion=512,
        environment_revision="abc123",
    )
    assert manifest["model"] == "org/model"
    assert manifest["benchmark"] == "org/benchmark"
    assert manifest["renderer"] == "native"
    assert manifest["environment_revision"] == "abc123"
    assert manifest["script_revision"]
    assert manifest["created_at_utc"]


def test_evaluators_have_explicit_main_entrypoints():
    for name in ("eval_bench_peft.py", "eval_hermes_env.py"):
        tree = ast.parse((SCRIPTS / name).read_text(encoding="utf-8"))
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        assert "main" in functions
        assert any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and any(isinstance(part, ast.Constant) and part.value == "__main__" for part in ast.walk(node))
            for node in tree.body
        )
