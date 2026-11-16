"""Shared contracts for the SakThai agentic evaluation and training scripts.

This module intentionally has no torch, transformers, datasets, or Hugging Face
imports. It can therefore be imported by CPU-only tests and by the lightweight
entrypoints without downloading a model or dataset.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

RENDER_MODES = ("handrolled", "native")
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

SYSTEM_CONTENT = (
    "You are a coding agent working in a sandboxed workspace. Use the "
    "available tools to complete the task, then call submit when you believe "
    "it is solved."
)

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run a shell command in the task workspace; returns combined stdout/stderr.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "Shell command to run."}}, "required": ["command"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the task workspace.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Relative path of the file to read."}}, "required": ["path"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite a file in the task workspace.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Relative path of the file to write."}, "content": {"type": "string", "description": "Full content to write to the file."}}, "required": ["path", "content"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patch",
            "description": "Find-and-replace a unique substring in a file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Relative path of the file to patch."}, "old_string": {"type": "string", "description": "Exact unique substring to replace."}, "new_string": {"type": "string", "description": "Replacement text."}}, "required": ["path", "old_string", "new_string"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "End the episode and grade the task. Call this only when you believe the task is solved.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def validate_choice(name: str, value: str, choices: Sequence[str]) -> str:
    """Validate and return a normalized string environment setting."""
    normalized = value.strip().lower()
    if normalized not in choices:
        raise ValueError(f"invalid {name}={value!r}; expected one of {list(choices)}")
    return normalized


def positive_int(name: str, value: str | int) -> int:
    """Parse a strictly positive integer setting."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero, got {parsed}")
    return parsed


def text(value: Any) -> str:
    return "" if value is None else value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def tools_block(tools: Sequence[Mapping[str, Any]] | None) -> str:
    if not tools:
        return ""
    signatures = "\n".join(json.dumps(tool, ensure_ascii=False) for tool in tools)
    return (
        "\n\n# Tools\n\nYou may call one or more functions. Signatures are within "
        "<tools></tools>:\n<tools>\n"
        + signatures
        + "\n</tools>\n\nFor each call return:\n"
        '<tool_call>\n{"name": <name>, "arguments": <json>}\n</tool_call>'
    )


def _assistant_body(message: Mapping[str, Any]) -> str:
    parts: list[str] = []
    content = text(message.get("content"))
    if content:
        parts.append(content)
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function", tool_call)
        arguments = function.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        parts.append(f'<tool_call>\n{{"name": "{function.get("name", "")}", "arguments": {arguments}}}\n</tool_call>')
    return "\n".join(parts)


def _render_message(message: Mapping[str, Any], tools: Sequence[Mapping[str, Any]] | None) -> str:
    role = message.get("role")
    if role == "system":
        return "<|im_start|>system\n" + text(message.get("content")) + tools_block(tools) + "<|im_end|>\n"
    if role == "user":
        return "<|im_start|>user\n" + text(message.get("content")) + "<|im_end|>\n"
    if role == "tool":
        return "<|im_start|>user\n<tool_response>\n" + text(message.get("content")) + "\n</tool_response><|im_end|>\n"
    if role == "assistant":
        return "<|im_start|>assistant\n" + _assistant_body(message) + "<|im_end|>\n"
    return ""


def render_prompt(messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] | None) -> str:
    """Render the hand-rolled benchmark-compatible ChatML prompt."""
    output: list[str] = []
    if not (messages and messages[0].get("role") == "system") and tools:
        output.append("<|im_start|>system\nYou are a helpful assistant." + tools_block(tools) + "<|im_end|>\n")
    for index, message in enumerate(messages):
        output.append(_render_message(message, tools if index == 0 and message.get("role") == "system" else None))
    output.append("<|im_start|>assistant\n")
    return "".join(output)


def parse_tool_calls(raw: str) -> list[dict[str, Any]]:
    """Parse all valid tagged tool calls from a model completion."""
    calls: list[dict[str, Any]] = []
    for match in TOOL_CALL_PATTERN.finditer(raw):
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if not data.get("name"):
            continue
        arguments = data.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"__unparsed__": arguments}
        calls.append({"name": data["name"], "arguments": arguments if isinstance(arguments, dict) else {}})
    return calls


def parse_tool_call(raw: str) -> dict[str, Any] | None:
    """Return the first valid tagged tool call, preserving legacy behavior."""
    calls = parse_tool_calls(raw)
    return calls[0] if calls else None


def git_revision() -> str | None:
    """Return the current git revision when running from a checkout."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("GITHUB_SHA")


def build_manifest(*, model: str, benchmark: str, renderer: str, max_steps: int | None, max_completion: int | None = None, environment_revision: str | None = None) -> dict[str, Any]:
    """Build stable provenance metadata for a result JSON document."""
    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_revision": git_revision(),
        "model": model,
        "benchmark": benchmark,
        "renderer": renderer,
        "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1") else "cpu-or-auto",
    }
    if max_steps is not None:
        manifest["max_steps"] = max_steps
    if max_completion is not None:
        manifest["max_completion"] = max_completion
    if environment_revision:
        manifest["environment_revision"] = environment_revision
    return manifest
