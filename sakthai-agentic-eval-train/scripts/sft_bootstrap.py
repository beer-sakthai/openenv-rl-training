# /// script
# dependencies = ["trl>=0.29.0", "openenv==0.4.1", "pydantic",
# "peft", "datasets", "accelerate", "transformers>=5.2.0",
# "huggingface_hub", "vllm"]


#                 "vllm"]
# ///
"""Stage 2: SFT bootstrap for sakthai-context-0.5b-tools.

The GRPO pilots proved the model solves the 6 hermes tasks 0/6, so GRPO got
zero reward signal (reward + gradient stayed 0 across 40 steps). This SFT step
teaches the model what a *successful* multi-step tool-use trajectory looks
like -- read the file(s), make the change, submit -- using the 6 oracle
trajectories already verified (Stage 1) to earn reward 1.0 against the env's
real check(). Goal: lift the model's success rate above 0 so a subsequent GRPO
run has reward variance to actually learn from.

LoRA (not full fine-tune) to (a) match the -tools family's adapter convention
and (b) limit distribution shift that could regress the model's strong
single-shot bench-v2 numbers (91.0% selection / 45.7% arguments). Re-eval on
BOTH hermes-env (did agentic go up?) and bench-v2 (did single-shot regress?)
after this.

Env vars:
  SFT_BASE       base checkpoint (default: Nanthasit/sakthai-context-0.5b-tools)
  SFT_ENV_REPO   hermes env repo (default: Nanthasit/hermes-tool-use-rl-env)
  SFT_REPEAT     copies of each of the 6 trajectories (default: 8 -> 48 rows)
  SFT_EPOCHS     training epochs (default: 3)
  SFT_PUSH_TO    repo to push the SFT adapter to (default: none)
"""

import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

BASE = os.environ.get("SFT_BASE", "Nanthasit/sakthai-context-0.5b-tools")
ENV_REPO = os.environ.get("SFT_ENV_REPO", "Nanthasit/hermes-tool-use-rl-env")
REPEAT = int(os.environ.get("SFT_REPEAT", "8"))
EPOCHS = float(os.environ.get("SFT_EPOCHS", "3"))
PUSH_TO = os.environ.get("SFT_PUSH_TO", "").strip()

env_dir = Path(snapshot_download(ENV_REPO, repo_type="dataset"))
sys.path.insert(0, str(env_dir))
sys.path.insert(0, str(env_dir / "server"))

from hermes_tool_env import HermesToolEnvironment  # noqa: E402
from models import HermesToolAction  # noqa: E402
from tasks import TASKS_BY_ID  # noqa: E402

SYSTEM = (
    "You are a coding agent working in a sandboxed workspace. Inspect the "
    "files, make the change the task asks for, then call submit to have it "
    "graded. Call exactly one tool per turn."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run a shell command in the task workspace; returns combined stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the task workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path of the file to read.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite a file in the task workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path of the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patch",
            "description": "Find-and-replace a unique substring in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path of the file to patch.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact unique substring to replace.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
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

ORACLES = {
    "fix_failing_test": [
        ("read_file", {"path": "calc.py"}),
        (
            "write_file",
            {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"},
        ),
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
        (
            "write_file",
            {
                "path": "helper.py",
                "content": "def new_name():\n    return 'hello world'\n",
            },
        ),
        (
            "write_file",
            {
                "path": "app.py",
                "content": "from helper import new_name\n\ndef main():\n    print(new_name())\n\n"
                "if __name__ == '__main__':\n    main()\n",
            },
        ),
        ("submit", {}),
    ],
    "fix_broken_imports": [
        ("read_file", {"path": "step1.py"}),
        (
            "write_file",
            {
                "path": "step1.py",
                "content": "from utils import double\nprint(double(3))\n",
            },
        ),
        (
            "write_file",
            {
                "path": "step2.py",
                "content": "from utils import double\nprint(double(21))\n",
            },
        ),
        ("submit", {}),
    ],
    "add_verbose_flag": [
        ("read_file", {"path": "greet.py"}),
        (
            "write_file",
            {
                "path": "greet.py",
                "content": "import sys\n\ndef main():\n    print('hello')\n    if '--verbose' in sys.argv:\n"
                "        print('DEBUG: verbose mode enabled')\n\nif __name__ == '__main__':\n    main()\n",
            },
        ),
        ("submit", {}),
    ],
    "count_log_errors": [
        ("read_file", {"path": "app.log"}),
        ("write_file", {"path": "error_count.txt", "content": "3"}),
        ("submit", {}),
    ],
}


def run_oracle(task_id):
    env = HermesToolEnvironment()
    env.reset()
    obs = env.step(HermesToolAction(tool="select_task", task_id=task_id))
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": obs.result},
    ]
    reward = 0.0
    for name, args in ORACLES[task_id]:
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {"type": "function", "function": {"name": name, "arguments": args}}
                ],
            }
        )
        obs = env.step(HermesToolAction(tool=name, **args))
        messages.append({"role": "tool", "name": name, "content": obs.result})
        if obs.done:
            reward = float(obs.reward or 0.0)
            break
    return reward, messages


def build_sft_dataset():
    from datasets import Dataset

    rows = []
    for task_id in sorted(TASKS_BY_ID):
        reward, messages = run_oracle(task_id)
        assert (
            reward >= 1.0
        ), f"oracle for {task_id} did not reach reward 1.0 (got {reward}) -- aborting SFT"
        for _ in range(REPEAT):
            rows.append({"messages": messages, "tools": TOOLS})
    return Dataset.from_list(rows)


def main():
    import torch
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    print(f"=== SFT bootstrap: base={BASE} repeat={REPEAT} epochs={EPOCHS} ===")
    dataset = build_sft_dataset()
    print(
        f"SFT dataset: {len(dataset)} rows (6 verified oracle trajectories x{REPEAT})"
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    args = SFTConfig(
        output_dir="./sft-bootstrap",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        max_length=2048,
        logging_steps=2,
        bf16=torch.cuda.is_available(),
        report_to=[],
        push_to_hub=bool(PUSH_TO),
        hub_model_id=PUSH_TO or None,
    )

    trainer = SFTTrainer(
        model=BASE,
        train_dataset=dataset,
        args=args,
        peft_config=peft_config,
    )

    t0 = time.time()
    trainer.train()
    print(f"=== SFT done in {time.time()-t0:.1f}s ===")

    if PUSH_TO:
        trainer.save_model("./sft-bootstrap-final")
        trainer.push_to_hub(
            commit_message=f"SFT bootstrap on 6 oracle hermes trajectories (x{REPEAT}, {EPOCHS} epochs)"
        )
        print(f"pushed -> {PUSH_TO}")
    else:
        print("SFT_PUSH_TO not set -- not pushing, mechanics/timing only")


if __name__ == "__main__":
    main()
