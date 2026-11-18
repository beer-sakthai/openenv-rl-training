"""Unit tests for sakthai-sft-training/scripts/eval/ data governance tools."""

import argparse
import hashlib
import json
import os
import sys

EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "eval"))
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

from contamination_audit import (
    _content_str,
    _first_user_prompt,
    _hash,
    _normalize,
    generate_report,
    get_training_specs,
)
from dataset_composition_stats import (
    _num_tool_calls,
    classify_row,
)


def test_contamination_audit_content_str():
    assert _content_str(None) == ""
    assert _content_str("hello world") == "hello world"
    d = {"b": 2, "a": 1}
    assert _content_str(d) == '{"a": 1, "b": 2}'


def test_contamination_audit_first_user_prompt():
    # JSON stringified messages list
    row_json = {
        "messages": json.dumps([
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Tell me a joke"},
            {"role": "assistant", "content": "Why did the chicken..."},
        ])
    }
    assert _first_user_prompt(row_json) == "Tell me a joke"

    # Python list of dicts
    row_list = {
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Solve 2+2"},
        ]
    }
    assert _first_user_prompt(row_list) == "Solve 2+2"

    # Bare prompt field fallback
    row_bare = {"prompt": "What is the capital of Thailand?"}
    assert _first_user_prompt(row_bare) == "What is the capital of Thailand?"

    # Bare query field fallback
    row_query = {"query": "Find flight to BKK"}
    assert _first_user_prompt(row_query) == "Find flight to BKK"

    # Missing or empty
    assert _first_user_prompt({}) == ""
    assert _first_user_prompt({"messages": "not json"}) == ""


def test_contamination_audit_normalize_and_hash():
    s1 = "  Hello \t World \n  "
    s2 = "hello world"
    assert _normalize(s1) == "hello world"
    assert _normalize(s2) == "hello world"
    assert _hash(_normalize(s1)) == hashlib.sha256(b"hello world").hexdigest()


def test_contamination_audit_get_training_specs():
    args = argparse.Namespace(
        training=["Nanthasit/sakthai-combined-v10", "Nanthasit/sakthai-combined-v11"],
        training_with_data_files=["Nanthasit/sakthai-combined-v12=data/train.jsonl"],
    )
    specs = get_training_specs(args)
    assert specs == [
        ("Nanthasit/sakthai-combined-v10", None),
        ("Nanthasit/sakthai-combined-v11", None),
        ("Nanthasit/sakthai-combined-v12", "data/train.jsonl"),
    ]


def test_contamination_audit_generate_report(tmp_path):
    report_file = tmp_path / "report.md"
    bench_idx = {
        "hash1": {"row_index": 0, "category": "coding", "prompt_preview": "print(1)"},
        "hash2": {"row_index": 1, "category": "chat", "prompt_preview": "hello"},
    }
    per_dataset_overlap = {
        "repoA": ["hash1"],
    }
    all_overlap_by_cat = {"coding": 1}
    total_overlap_hashes = {"hash1"}

    generate_report(
        bench="test-bench",
        bench_idx=bench_idx,
        per_dataset_overlap=per_dataset_overlap,
        all_overlap_by_cat=all_overlap_by_cat,
        total_overlap_hashes=total_overlap_hashes,
        out=str(report_file),
    )

    content = report_file.read_text()
    assert "# Contamination audit — test-bench" in content
    assert "Distinct bench prompts also present in the training mix: **1** (50.0%)" in content
    assert "`repoA`" in content


def test_dataset_composition_num_tool_calls():
    assert _num_tool_calls("") == 0
    assert _num_tool_calls("regular response without tool calls") == 0
    assert _num_tool_calls("<tool_call>{\"name\": \"calc\"}</tool_call>") == 1
    # Multiple and case-insensitive
    text = "<tool_call>{\"name\": \"a\"}</tool_call> and <TOOL_CALL>{\"name\": \"b\"}</tool_call>"
    assert _num_tool_calls(text) == 2


def test_dataset_composition_classify_row():
    # Irrelevance: no tool calls
    row_irrelevant = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello there!"},
        ]
    }
    assert classify_row(row_irrelevant) == "irrelevance"

    # Simple: exactly 1 assistant turn with 1 tool call
    row_simple = {
        "messages": [
            {"role": "user", "content": "calc 2+2"},
            {"role": "assistant", "content": "<tool_call>{\"name\": \"calc\"}</tool_call>"},
        ]
    }
    assert classify_row(row_simple) == "simple"

    # Parallel: 1 assistant turn with >1 tool call
    row_parallel = {
        "messages": [
            {"role": "user", "content": "search two things"},
            {"role": "assistant", "content": "<tool_call>search 1</tool_call><tool_call>search 2</tool_call>"},
        ]
    }
    assert classify_row(row_parallel) == "parallel"

    # Multi-turn: 2+ assistant turns with tool calls
    row_multiturn = {
        "messages": [
            {"role": "user", "content": "step 1"},
            {"role": "assistant", "content": "<tool_call>call 1</tool_call>"},
            {"role": "tool", "content": "result 1"},
            {"role": "assistant", "content": "<tool_call>call 2</tool_call>"},
        ]
    }
    assert classify_row(row_multiturn) == "multi_turn"

    # Other: invalid or empty structure
    assert classify_row({}) == "other"
    assert classify_row({"messages": "invalid json"}) == "other"
