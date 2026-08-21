# /// script
# dependencies = [
#   "torch",
#   "transformers>=5.2.0",
#   "datasets",
#   "trl",
#   "peft",
#   "accelerate",
#   "jmespath",
#   # --env browsergym needs BrowserGymEnv from the OpenEnv monorepo (not on PyPI):
#   "browsergym_env @ git+https://github.com/huggingface/OpenEnv.git#subdirectory=envs/browsergym_env",
# ]
# ///

"""GRPO training entrypoint for custom environments.

    python train.py --env simple                            # Tier A, inline, no server
    python train.py --env agent_tools --vllm-mode colocate  # Tier B, sandbox server must be up
    python train.py --env browsergym                        # BrowserGym MiniWoB++ (Nanthasit Space)
    python train.py --env browsergym --browsergym-task email-inbox  # harder task

Every CLI flag also accepts a TRAIN_* environment variable so the
sakthai-jobs-dispatcher can drive this script without translating env
vars into CLI args at submit time.

Model is configurable (--model) - defaults are task-appropriate. See README.md
for model-size guidance and the whole-episode max_completion_length caveat.

BrowserGym Space: https://huggingface.co/spaces/Nanthasit/browsergym-env
BrowserGym URL:   https://nanthasit-browsergym-env.hf.space
"""

import argparse
import os

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

# Default BrowserGym Space URL (Nanthasit-owned)
BROWSERGYM_SPACE_URL = "https://nanthasit-browsergym-env.hf.space"


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _make_browsergym_factory(task_name: str, space_url: str):
    """Build a GRPOTrainer-compatible environment factory for BrowserGym."""
    def factory():
        try:
            from browsergym_env import BrowserGymEnv
            return BrowserGymEnv(
                base_url=space_url,
                environment={
                    "BROWSERGYM_BENCHMARK": "miniwob",
                    "BROWSERGYM_TASK_NAME": task_name,
                    "BROWSERGYM_HEADLESS": "true",
                },
            )
        except ImportError:
            raise ImportError(
                "browsergym_env not installed. "
                "Run: pip install git+https://github.com/huggingface/OpenEnv.git"
            )
    return factory


def _browsergym_reward(completions, **kwargs):
    """Reward function: pass BrowserGym step reward back to GRPO."""
    rewards = []
    for env_output in kwargs.get("env_outputs", []):
        rewards.append(float(env_output.get("reward", 0.0)))
    if not rewards:
        rewards = [0.0] * len(completions)
    return rewards


def _browsergym_dataset(n_episodes: int):
    """Minimal prompt dataset for BrowserGym web navigation tasks."""
    prompts = [
        {"prompt": "You are a web navigation agent. Complete the task shown in the browser. "
                   "Use click(), fill(), goto(), press(), scroll() actions. "
                   "Observe the page carefully and act step by step."}
    ] * n_episodes
    return Dataset.from_list(prompts)


BROWSERGYM_DEFAULT_MODEL = "Nanthasit/sakthai-context-7b-tools"


def _select(args: argparse.Namespace):
    """Return (factory, reward_funcs, dataset_builder, default_model)."""
    if args.env == "simple":
        from env_simple_task import (
            DEFAULT_MODEL, ENVIRONMENT_FACTORY, REWARD_FUNCS, TRAIN_DATASET,
        )
    elif args.env == "agent_tools":
        from agent_tools.wrapper import (
            DEFAULT_MODEL, ENVIRONMENT_FACTORY, REWARD_FUNCS, TRAIN_DATASET,
        )
    elif args.env == "browsergym":
        task = getattr(args, "browsergym_task", "click-test")
        url  = getattr(args, "browsergym_url", BROWSERGYM_SPACE_URL)
        ENVIRONMENT_FACTORY = _make_browsergym_factory(task, url)
        REWARD_FUNCS        = [_browsergym_reward]
        TRAIN_DATASET       = _browsergym_dataset
        DEFAULT_MODEL       = BROWSERGYM_DEFAULT_MODEL
    else:
        raise SystemExit(f"unknown --env {args.env!r}; choose 'simple', 'agent_tools', or 'browsergym'")
    return ENVIRONMENT_FACTORY, REWARD_FUNCS, TRAIN_DATASET, DEFAULT_MODEL


def main():
    env_default = os.environ.get("TRAIN_ENV")
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["simple", "agent_tools", "browsergym"],
                        default=env_default, required=(env_default is None))
    parser.add_argument("--browsergym-task",
                        default=os.environ.get("TRAIN_BROWSERGYM_TASK", "click-test"),
                        help="MiniWoB task name (default: click-test). e.g. click-button, email-inbox")
    parser.add_argument("--browsergym-url",
                        default=os.environ.get("TRAIN_BROWSERGYM_URL", BROWSERGYM_SPACE_URL),
                        help=f"BrowserGym Space URL (default: {BROWSERGYM_SPACE_URL})")
    parser.add_argument("--model",
                        default=os.environ.get("TRAIN_BASE"),
                        help="defaults to Nanthasit/sakthai-context-7b-tools (or task default). Env: TRAIN_BASE")
    parser.add_argument("--vllm-mode", choices=["colocate", "server"],
                        default=os.environ.get("TRAIN_VLLM_MODE", "colocate"))
    parser.add_argument("--vllm-server-host",
                        default=os.environ.get("TRAIN_VLLM_HOST", "localhost"))
    parser.add_argument("--vllm-server-port", type=int,
                        default=int(os.environ.get("TRAIN_VLLM_PORT", "8000")))
    # Caps tokens across the WHOLE multi-turn episode (generations + tool
    # results summed), not one turn - raise if episodes truncate mid-task.
    parser.add_argument("--max-completion-length", type=int,
                        default=int(os.environ.get("TRAIN_MAX_COMPLETION", "1024")))
    parser.add_argument("--num-generations", type=int,
                        default=int(os.environ.get("TRAIN_NUM_GENERATIONS", "4")))
    parser.add_argument("--gradient-accumulation-steps", type=int,
                        default=int(os.environ.get("TRAIN_GRAD_ACCUM", "64")))
    parser.add_argument("--n-episodes", type=int,
                        default=int(os.environ.get("TRAIN_EPISODES", "64")))
    parser.add_argument("--enable-thinking", action="store_true",
                        default=_bool_env("TRAIN_ENABLE_THINKING", False),
                        help="flip on for harder tasks; costs more tokens/turn. "
                             "Only Qwen3-template bases act on this; Qwen2 templates ignore it.")
    parser.add_argument("--push-to-hub",
                        default=os.environ.get("TRAIN_PUSH_TO"),
                        help="repo id to push the trained model to (optional). Env: TRAIN_PUSH_TO")
    args = parser.parse_args()

    factory, rewards, dataset_builder, default_model = _select(args)
    model = args.model or default_model
    dataset: Dataset = dataset_builder(args.n_episodes)

    grpo_kwargs = dict(
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        chat_template_kwargs={"enable_thinking": args.enable_thinking},
        log_completions=True,
    )
    if args.vllm_mode == "server":
        grpo_kwargs["vllm_server_host"] = args.vllm_server_host
        grpo_kwargs["vllm_server_port"] = args.vllm_server_port

    trainer = GRPOTrainer(
        model=model,
        train_dataset=dataset,
        reward_funcs=rewards,
        args=GRPOConfig(**grpo_kwargs),
        environment_factory=factory,
    )
    trainer.train()

    if args.push_to_hub:
        trainer.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
