# /// script
# dependencies = ["trl>=0.29.0", "openenv==0.4.1", "jmespath",
# "peft", "datasets", "accelerate", "transformers>=5.2.0", "vllm",
# "huggingface_hub"]


#                 "huggingface_hub"]
# ///
"""GRPO training pilot for sakthai-context-0.5b-tools, agentic tool-use fix.

Continues from the existing Nanthasit/sakthai-context-0.5b-tools checkpoint
(already strong on single-shot tool selection per sakthai-bench-v2: 91.0%
selection / 45.7% arguments) using hermes-tool-use-rl-env's 6-task agentic
coding environment as the GRPO reward signal -- the same environment the
eval harness already proved this checkpoint scores 0/6 on.

This is a SMALL, BOUNDED PILOT, not a production run: small dataset
(episodes_per_task default 4, not train.py's default 32), few optimizer
steps (TRAIN_MAX_STEPS, default 6), and it pushes its result to a clearly
labeled -grpo-pilot-* repo, not over the real -tools repo. Purpose: get
real per-step timing/cost and confirm reward is directionally moving before
committing to a real (larger, longer) run.

Fixes applied vs. the original train.py in Nanthasit/hermes-tool-use-rl-env
(see investigation notes):
  - environment_factory points at HermesToolEnvironment directly (in-process),
    not the broken client.py/Docker/WebSocket path.
  - vllm_mode="colocate" (train.py's CLI default, --vllm-mode server, passes
    a nonexistent vllm_server_url GRPOConfig field (now fixed) and crashed at construction).
  - jmespath added as an explicit dependency (GRPOTrainer hard-requires it
    for tool-call parsing when environment_factory/tools is used; not listed
    in the original repo's README install line).

Usage (HF Jobs):
    hf jobs uv run --flavor l4x1 --secrets HF_TOKEN --timeout 30m \
      --env TRAIN_MODE=lora16 \
      --env TRAIN_PUSH_TO=Nanthasit/sakthai-context-0.5b-tools-grpo-pilot-lora16 \
      ./grpo_train_pilot.py

Env vars:
  TRAIN_MODE        lora16 | lora64 | full  (required)
  TRAIN_BASE        starting checkpoint (default: Nanthasit/sakthai-context-0.5b-tools)
  TRAIN_ENV_REPO    hermes-tool-use-rl-env dataset repo (default: Nanthasit/hermes-tool-use-rl-env)
  TRAIN_EPISODES    episodes per task for the pilot dataset (default: 4 -> 24 rows for 6 tasks)
  TRAIN_MAX_STEPS   optimizer steps to run before stopping (default: 6)
  TRAIN_PUSH_TO     repo to push the trained adapter/model to (optional; if unset, does not push)
"""

import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

MODE = os.environ.get("TRAIN_MODE", "").strip()
assert MODE in ("lora16", "lora64", "full"), "set TRAIN_MODE to lora16, lora64, or full"
BASE_MODEL = os.environ.get("TRAIN_BASE", "Nanthasit/sakthai-context-0.5b-tools")
ENV_REPO = os.environ.get("TRAIN_ENV_REPO", "Nanthasit/hermes-tool-use-rl-env")
EPISODES_PER_TASK = int(os.environ.get("TRAIN_EPISODES", "4"))
MAX_STEPS = int(os.environ.get("TRAIN_MAX_STEPS", "6"))
# Caps tokens across the WHOLE multi-turn episode (every generation + every tool
# result). 512 is too tight for the multi-step tasks (up to 12 tool calls) -- a
# too-small cap guarantees reward 0, which starves GRPO of any learning signal.
MAX_COMPLETION = int(os.environ.get("TRAIN_MAX_COMPLETION", "512"))
PUSH_TO = os.environ.get("TRAIN_PUSH_TO", "").strip()

# ── Fetch the environment source and import it directly (bypasses the
#    broken client.py/Docker path entirely, same approach as the eval harness) ──
env_dir = Path(snapshot_download(ENV_REPO, repo_type="dataset"))
sys.path.insert(0, str(env_dir))
sys.path.insert(0, str(env_dir / "server"))

from hermes_tool_env import HermesToolEnvironment  # noqa: E402
from models import HermesToolAction  # noqa: E402
from tasks import TASKS, TASKS_BY_ID  # noqa: E402

TASK_IDS = sorted(TASKS_BY_ID)


class HermesToolTaskEnvLocal:
    """Drop-in replacement for trl_env.py's HermesToolTaskEnv: same
    reset(**kwargs)->Optional[str] / auto-registered-methods-as-tools
    contract that trl's EnvironmentFactory protocol expects, but talks to
    HermesToolEnvironment in-process instead of over the broken client.py.

    Tool-method docstrings are the exact Google-style Args:-block form from
    trl_env.py -- trl builds each tool's JSON schema from these via
    transformers.get_json_schema, which raises DocstringParsingException if
    any parameter is missing an Args: description. One-line docstrings fail.
    """

    def __init__(self):
        self._env = HermesToolEnvironment()
        self.reward = 0.0
        self.done = False

    def reset(self, **kwargs):
        self._env = HermesToolEnvironment()
        self._env.reset()
        self.reward = 0.0
        self.done = False
        task_id = kwargs.get("task")
        if task_id and task_id in TASKS_BY_ID:
            obs = self._env.step(HermesToolAction(tool="select_task", task_id=task_id))
        else:
            obs = self._env.step(
                HermesToolAction(tool="select_task", task_id=TASK_IDS[0])
            )
        return obs.result

    def terminal(self, command: str) -> str:
        """Run a shell command in the task workspace and return its output.

        Args:
            command: Shell command to execute. Runs with a short timeout;
                long-running or interactive commands will be killed.

        Returns:
            Combined stdout/stderr from the command.
        """
        return self._step(HermesToolAction(tool="terminal", command=command))

    def read_file(self, path: str) -> str:
        """Read a file from the task workspace.

        Args:
            path: Path relative to the workspace root.

        Returns:
            The file's contents, or an error message if it doesn't exist.
        """
        return self._step(HermesToolAction(tool="read_file", path=path))

    def write_file(self, path: str, content: str) -> str:
        """Write (overwrite) a file in the task workspace.

        Args:
            path: Path relative to the workspace root.
            content: Full contents to write.

        Returns:
            A confirmation or error message.
        """
        return self._step(
            HermesToolAction(tool="write_file", path=path, content=content)
        )

    def patch(self, path: str, old_string: str, new_string: str) -> str:
        """Find-and-replace a unique substring in a file.

        Args:
            path: Path relative to the workspace root.
            old_string: Exact text to find; must appear exactly once in the file.
            new_string: Text to replace it with.

        Returns:
            A confirmation, or an error if old_string isn't found or isn't unique.
        """
        return self._step(
            HermesToolAction(
                tool="patch", path=path, old_string=old_string, new_string=new_string
            )
        )

    def submit(self) -> str:
        """Signal that the task is complete and end the episode. Call this
        once you believe the task's requirements are satisfied -- the episode
        is graded at this point and no further tool calls are possible after.

        Returns:
            Whether the task passed grading.
        """
        return self._step(HermesToolAction(tool="submit"))

    def _step(self, action: HermesToolAction) -> str:
        if self.done:
            return "Episode already ended."
        obs = self._env.step(action)
        self.reward = float(obs.reward or 0.0)
        self.done = bool(obs.done)
        return obs.result


def build_dataset(episodes_per_task):
    """One row per (task, episode) -- mirrors train.py's build_dataset(), just
    smaller for a pilot. 'task' is read by HermesToolTaskEnvLocal.reset(**kwargs)
    to pin which of the 6 TaskSpecs this rollout is graded against.

    NOTE: 'prompt' MUST be conversational (a list of {role, content} message
    dicts), not a plain string -- trl's tool-calling GRPO does prompt[-1]["content"]
    to read the last message, so a plain string fails with
    "TypeError: string indices must be integers". This matches train.py line 40.
    """
    from datasets import Dataset

    prompts = []
    task_ids = []
    for task in TASKS:
        for _ in range(episodes_per_task):
            prompts.append([{"role": "user", "content": task.prompt}])
            task_ids.append(task.task_id)
    return Dataset.from_dict({"prompt": prompts, "task": task_ids})


def reward_func(environments, **kwargs):
    """Thin pass-through of HermesToolEnvironment's binary task-pass reward,
    identical in shape to the original train.py's reward_func.
    """
    return [env.reward for env in environments]


def _peft_base_model(repo_id):
    """Return base_model_name_or_path if repo_id is a bare PEFT adapter, else None."""
    try:
        from huggingface_hub import hf_hub_download

        cfg_path = hf_hub_download(repo_id, "adapter_config.json")
    except Exception:
        return None
    import json

    with open(cfg_path) as f:
        cfg = json.load(f)
    return cfg.get("base_model_name_or_path")


def resolve_model_dir(repo_id):
    """vLLM colocate can't train from a bare adapter repo -- it needs a full
    model dir. If repo_id is a PEFT adapter, merge it into its base and save a
    full model to a local dir (both HF + vLLM load from a dir). Else return the
    repo_id unchanged (already a full model)."""
    base_id = _peft_base_model(repo_id)
    if base_id is None:
        return repo_id
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  merging adapter {repo_id} into base {base_id} -> ./merged_base ...")
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, repo_id).merge_and_unload()
    merged.save_pretrained("./merged_base")
    try:
        tok = AutoTokenizer.from_pretrained(repo_id)
    except Exception:
        tok = AutoTokenizer.from_pretrained(base_id)
    tok.save_pretrained("./merged_base")
    del base, merged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "./merged_base"


def get_peft_config(mode):
    if mode == "lora16":
        from peft import LoraConfig

        return LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
    elif mode == "lora64":
        from peft import LoraConfig

        return LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
    return None

def get_grpo_config(mode, max_completion, max_steps):
    import torch
    from trl import GRPOConfig

    return GRPOConfig(
        output_dir=f"./grpo-pilot-{mode}",
        use_vllm=True,
        vllm_mode="colocate",  # NOT "server" -- avoids older train.py vllm_server_url bug
        max_completion_length=max_completion,
        num_generations=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        max_steps=max_steps,
        chat_template_kwargs={"enable_thinking": False},
        log_completions=False,  # rich-console pretty-printer can crash on non-UTF-8 consoles; keep off for jobs
        bf16=torch.cuda.is_available(),
        report_to=[],
        # Don't use trainer's auto-push: with a merged-local base, the pushed
        # adapter's base_model_name_or_path would be "./merged_base" (a broken
        # reference). Instead we merge the GRPO adapter back out and push a
        # standalone full model below.
        push_to_hub=False,
    )

def push_model_to_hub(trainer, model_path, mode, base_model, max_steps, push_to):
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    out_dir = f"./grpo-pilot-{mode}-final"
    model_to_save = trainer.model
    if hasattr(model_to_save, "merge_and_unload"):
        print("  merging GRPO LoRA into base for a standalone push ...")
        model_to_save = model_to_save.merge_and_unload()
    model_to_save.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(model_path).save_pretrained(out_dir)
    HfApi().create_repo(push_to, exist_ok=True)
    HfApi().upload_folder(
        folder_path=out_dir,
        repo_id=push_to,
        commit_message=f"GRPO pilot ({mode}) from {base_model}, {max_steps} steps",
    )
    print(f"pushed -> {push_to}")

def main():
    from trl import GRPOTrainer

    print(
        f"=== GRPO pilot: mode={MODE} base={BASE_MODEL} episodes/task={EPISODES_PER_TASK} max_steps={MAX_STEPS} ==="
    )

    # Merge-in: adapter repos -> local full-model dir so vLLM colocate can load it.
    model_path = resolve_model_dir(BASE_MODEL)
    print(f"  training from: {model_path}")

    dataset = build_dataset(EPISODES_PER_TASK)
    print(f"dataset: {len(dataset)} rows")

    peft_config = get_peft_config(MODE)
    args = get_grpo_config(MODE, MAX_COMPLETION, MAX_STEPS)

    trainer = GRPOTrainer(
        model=model_path,
        train_dataset=dataset,
        reward_funcs=reward_func,
        args=args,
        peft_config=peft_config,
        environment_factory=HermesToolTaskEnvLocal,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(
        f"=== training done in {elapsed:.1f}s ({elapsed / max(MAX_STEPS, 1):.1f}s/step) ==="
    )

    if PUSH_TO:
        push_model_to_hub(trainer, model_path, MODE, BASE_MODEL, MAX_STEPS, PUSH_TO)
    else:
        print("TRAIN_PUSH_TO not set -- not pushing, pilot mechanics/timing only")


if __name__ == "__main__":
    main()
