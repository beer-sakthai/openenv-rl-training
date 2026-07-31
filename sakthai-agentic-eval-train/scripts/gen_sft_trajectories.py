# /// script
# dependencies = ["openenv==0.4.1", "pydantic", "huggingface_hub"]
# ///
"""Stage 1: generate + verify oracle SFT trajectories for the 6 hermes tasks.

Purpose: sakthai-context-0.5b-tools solves these tasks 0/6, so GRPO gets no
reward signal (see the failed GRPO pilots -- reward stayed 0, gradient 0).
The fix is an SFT bootstrap: teach the model what a *successful* multi-step
tool-use trajectory looks like, lifting its success rate above 0 so GRPO then
has something to reinforce.

This script drives HermesToolEnvironment in-process (no Docker, same as the
eval harness), replaying a hand-written ORACLE solution for each task, and:
  1. verifies every oracle actually earns reward 1.0 via the env's real
     check() -- if an oracle is wrong, you'll see reward 0.0 and a FAIL here,
     BEFORE any GPU is spent on SFT.
  2. records each successful episode as standard conversational tool-use SFT
     data ({"messages": [...], "tools": [...]}), the format Qwen2.5's chat
     template renders as <tool_call>{"name":...,"arguments":...}</tool_call>
     -- matching how the model is prompted at eval/GRPO time.

Stage 1 does NOT push anything -- it prints the trajectories + a reward
summary for inspection. Once all 6 show reward 1.0, Stage 2 regenerates
(deterministic) and runs the actual SFT.

Env vars:
  SAK_ENV_REPO   hermes env repo (default: Nanthasit/hermes-tool-use-rl-env)
  SAK_REPEAT     copies of each trajectory to emit (default: 1; SFT stage bumps this)
  SAK_PUSH_TO    if set, upload the JSONL dataset to this dataset repo (default: none)
"""
import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ENV_REPO = os.environ.get("SAK_ENV_REPO", "Nanthasit/hermes-tool-use-rl-env")
REPEAT = int(os.environ.get("SAK_REPEAT", "1"))
PUSH_TO = os.environ.get("SAK_PUSH_TO", "").strip()

env_dir = Path(snapshot_download(ENV_REPO, repo_type="dataset"))
sys.path.insert(0, str(env_dir))
sys.path.insert(0, str(env_dir / "server"))

from models import HermesToolAction  # noqa: E402
from tasks import TASKS_BY_ID  # noqa: E402
from hermes_tool_env import HermesToolEnvironment  # noqa: E402

SYSTEM = ("You are a coding agent working in a sandboxed workspace. Inspect the "
          "files, make the change the task asks for, then call submit to have it "
          "graded. Call exactly one tool per turn.")

# Tool schemas the model sees (the 5 real Hermes tools; select_task is env-internal).
TOOLS = [
    {"type": "function", "function": {"name": "terminal",
        "description": "Run a shell command in the task workspace; returns combined stdout/stderr.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Shell command to run."}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file",
        "description": "Read a file from the task workspace.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative path of the file to read."}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file",
        "description": "Write or overwrite a file in the task workspace.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative path of the file to write."},
            "content": {"type": "string", "description": "Full content to write to the file."}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "patch",
        "description": "Find-and-replace a unique substring in a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative path of the file to patch."},
            "old_string": {"type": "string", "description": "Exact unique substring to replace."},
            "new_string": {"type": "string", "description": "Replacement text."}},
            "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {"name": "submit",
        "description": "End the episode and grade the task. Call this only when you believe the task is solved.",
        "parameters": {"type": "object", "properties": {}}}},
]

# ── Oracle solutions: each a list of (tool_name, arguments) tool calls that,
#    replayed in order, drive the task to a passing check() state. Derived
#    directly from tasks.py's real setup()/check() logic, so they are provably
#    correct -- and the script verifies reward==1.0 at runtime regardless.
ORACLES = {
    "fix_failing_test": [
        ("read_file", {"path": "calc.py"}),
        ("write_file", {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"}),
        ("submit", {}),
    ],
    "extract_and_summarize": [
        ("read_file", {"path": "data.csv"}),
        ("write_file", {"path": "summary.txt", "content": "rows: 4\ntotal: 100\n"}),
        ("submit", {}),
    ],
    "rename_across_files": [
        ("read_file", {"path": "app.py"}),
        ("read_file", {"path": "helper.py"}),
        ("write_file", {"path": "helper.py", "content": "def new_name():\n    return 'hello world'\n"}),
        ("write_file", {"path": "app.py", "content":
            "from helper import new_name\n\ndef main():\n    print(new_name())\n\n"
            "if __name__ == '__main__':\n    main()\n"}),
        ("submit", {}),
    ],
    "fix_broken_imports": [
        ("read_file", {"path": "step1.py"}),
        ("write_file", {"path": "step1.py", "content": "from utils import double\nprint(double(3))\n"}),
        ("write_file", {"path": "step2.py", "content": "from utils import double\nprint(double(21))\n"}),
        ("submit", {}),
    ],
    "add_verbose_flag": [
        ("read_file", {"path": "greet.py"}),
        ("write_file", {"path": "greet.py", "content":
            "import sys\n\ndef main():\n    print('hello')\n    if '--verbose' in sys.argv:\n"
            "        print('DEBUG: verbose mode enabled')\n\nif __name__ == '__main__':\n    main()\n"}),
        ("submit", {}),
    ],
    "count_log_errors": [
        ("read_file", {"path": "app.log"}),
        ("write_file", {"path": "error_count.txt", "content": "3"}),
        ("submit", {}),
    ],
}


def run_oracle(task_id):
    """Replay the oracle for one task through the real env; return (reward, messages)."""
    env = HermesToolEnvironment()
    env.reset()
    obs = env.step(HermesToolAction(tool="select_task", task_id=task_id))

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": obs.result},
    ]
    reward = 0.0
    for name, args in ORACLES[task_id]:
        messages.append({"role": "assistant", "tool_calls": [
            {"type": "function", "function": {"name": name, "arguments": args}}]})
        obs = env.step(HermesToolAction(tool=name, **args))
        messages.append({"role": "tool", "name": name, "content": obs.result})
        if obs.done:
            reward = float(obs.reward or 0.0)
            break
    return reward, messages


def main():
    all_rows = []
    summary = {}
    for task_id in sorted(TASKS_BY_ID):
        reward, messages = run_oracle(task_id)
        summary[task_id] = reward
        status = "PASS" if reward >= 1.0 else "FAIL"
        print(f"\n{'='*70}\n[{status}] {task_id}: reward={reward}\n{'='*70}")
        for m in messages:
            if m["role"] == "assistant":
                tc = m["tool_calls"][0]["function"]
                print(f"  assistant -> {tc['name']}({json.dumps(tc['arguments'])[:120]})")
            else:
                print(f"  {m['role']}: {str(m.get('content',''))[:100]!r}")
        if reward >= 1.0:
            for _ in range(REPEAT):
                all_rows.append({"messages": messages, "tools": TOOLS})

    print(f"\n{'#'*70}")
    passed = sum(1 for r in summary.values() if r >= 1.0)
    print(f"# ORACLE VERIFICATION: {passed}/{len(summary)} tasks reach reward 1.0")
    for tid, r in summary.items():
        print(f"#   {tid}: {r}")
    print(f"# emitted {len(all_rows)} SFT rows ({REPEAT} copies each of {passed} passing tasks)")
    print(f"{'#'*70}")

    if passed < len(summary):
        print("!! Not all oracles pass -- fix the failing oracle(s) before SFT. Not pushing.")
        sys.exit(1)

    out_path = "hermes_sft_trajectories.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out_path} ({len(all_rows)} rows)")

    if PUSH_TO:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=out_path, path_in_repo="data/train.jsonl",
                            repo_id=PUSH_TO, repo_type="dataset",
                            commit_message=f"Oracle SFT trajectories, {len(all_rows)} rows, 6/6 tasks verified reward 1.0")
        print(f"pushed -> {PUSH_TO}")


if __name__ == "__main__":
    main()
