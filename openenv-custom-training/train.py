"""GRPO training entrypoint for the two custom environments.

    python train.py --env simple                       # Tier A, inline, no server
    python train.py --env agent_tools --vllm-mode colocate   # Tier B, sandbox server must be up

Model is configurable (--model) — defaults are task-appropriate. See README.md
for model-size guidance and the whole-episode max_completion_length caveat.
"""

import argparse

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer


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
    else:
        raise SystemExit(f"unknown --env {args.env!r}; choose 'simple' or 'agent_tools'")
    return ENVIRONMENT_FACTORY, REWARD_FUNCS, TRAIN_DATASET, DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["simple", "agent_tools"], required=True)
    parser.add_argument("--model", default=None,
                        help="defaults to Nanthasit/sakthai-context-7b-tools (or task default)")
    parser.add_argument("--vllm-mode", choices=["colocate", "server"], default="colocate")
    parser.add_argument("--vllm-server-host", default="localhost")
    parser.add_argument("--vllm-server-port", type=int, default=8000)
    # Caps tokens across the WHOLE multi-turn episode (generations + tool
    # results summed), not one turn — raise if episodes truncate mid-task.
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=64)
    parser.add_argument("--n-episodes", type=int, default=64)
    parser.add_argument("--enable-thinking", action="store_true",
                        help="flip on for harder tasks; costs more tokens/turn. "
                             "Only Qwen3-template bases act on this; Qwen2 templates ignore it.")
    parser.add_argument("--push-to-hub", default=None,
                        help="repo id to push the trained model to (optional)")
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
        reward_funcs=rewards,          # one func (or a list, as in multi_env.py)
        args=GRPOConfig(**grpo_kwargs),
        environment_factory=factory,   # the CLASS, not an instance
    )
    trainer.train()

    if args.push_to_hub:
        trainer.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
