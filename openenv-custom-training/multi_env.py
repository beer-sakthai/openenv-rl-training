"""One model trained across BOTH custom environments in a single GRPO run:
the inline Tier A game and the sandboxed Tier B agent-tools task.

Uses TRL 1.9.2's NATIVE multi-environment form (verified against the installed
source): pass `environment_factory` as a dict mapping a name -> environment
class. Each dataset row carries an `environment` column naming which factory
the episode uses; the `environment` column is a control field (NOT forwarded
to reset), and every OTHER column is passed to `reset(**kwargs)`.

Benefits over the skill's meta-environment class (one class owning N
sub-environments): each episode is rendered with ONLY its environment's tools
in the schema (no cross-task tool pollution), and reward funcs discriminate by
exact class type. The meta-class pattern from
references/multi-environment.md still works if you prefer it — the per-env
reward functions below follow that doc's "None for other-env episodes" rule.

Reward: one function per environment, each returning `None` for episodes that
belonged to the other (TRL turns None into NaN and aggregates each reward over
its own episodes only). Also note: TRL treats an env method named `get_reward`
as a framework-native reward source (auto-registered, exact-type routed) — an
alternative to reward_funcs, not combined with it.

Requires the Tier B sandbox server to be up (agent_tools/README.md).

Watch `train/reward_func_0` (guess) and `train/reward_func_1` (agent_tools)
individually — the combined `train/reward` alternates between tasks batch to
batch and reads as noisy even when training is healthy.
"""

import argparse
from typing import Optional

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from env_simple_task import MAX_ATTEMPTS, SimpleGuessEnv
from agent_tools.wrapper import AgentToolEnv, TASK_KEYS, TASK_PROMPTS

TASKS_PER_ENV_DEFAULT = 32


# ============================================================================
# Reward functions — one per environment, None for episodes that aren't its own.
# In the dict-factory form every reward func receives the whole batch's mixed
# instances, so discriminate by exact class type (mirrors how TRL's own
# get_reward routing works).
# ============================================================================

def guess_reward(environments, **kwargs) -> list[Optional[float]]:
    return [env.reward if type(env) is SimpleGuessEnv else None for env in environments]


def agent_tools_reward(environments, **kwargs) -> list[Optional[float]]:
    return [env.reward if type(env) is AgentToolEnv else None for env in environments]


REWARD_FUNCS = [guess_reward, agent_tools_reward]


# ============================================================================
# Dataset — both tasks, tagged with the routing column. The control column is
# named `environment` (TRL's convention for dict factories); each row carries
# only its own env's reset kwargs (`target` for guess, `task` for agent_tools).
# ============================================================================

def build_dataset(n_per_env: int = TASKS_PER_ENV_DEFAULT) -> Dataset:
    rows, environments, targets, tasks = [], [], [], []
    for i in range(n_per_env):
        # guess row
        target = (i * 7) % 100
        rows.append([{
            "role": "user",
            "content": (
                "I'm thinking of a number between 0 and 99. Use the guess(number) "
                f"tool to find it using higher/lower feedback ({MAX_ATTEMPTS} attempts)."
            ),
        }])
        environments.append("guess")
        targets.append(target)
        tasks.append(None)
        # agent_tools row
        task_key = TASK_KEYS[i % len(TASK_KEYS)]
        rows.append([{"role": "user", "content": TASK_PROMPTS[task_key]}])
        environments.append("agent_tools")
        targets.append(None)
        tasks.append(task_key)
    return Dataset.from_dict({
        "prompt": rows,
        "environment": environments,
        "target": targets,
        "task": tasks,
    })


# ============================================================================
# Training
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Nanthasit/sakthai-context-1.5b-merged",
                        help="one SakThai-family base learns both tasks; swap any model via --model")
    parser.add_argument("--vllm-mode", choices=["colocate", "server"], default="colocate")
    parser.add_argument("--vllm-server-url", default="http://localhost:8000")
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=64)
    parser.add_argument("--n-per-env", type=int, default=TASKS_PER_ENV_DEFAULT)
    args = parser.parse_args()

    grpo_kwargs = dict(
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # enable_thinking is a Qwen3-template kwarg; the SakThai Qwen2 base
        # ignores it (harmless) and it matters only if you swap in a Qwen3 model.
        chat_template_kwargs={"enable_thinking": False},
        log_completions=True,
    )
    if args.vllm_mode == "server":
        grpo_kwargs["vllm_server_url"] = args.vllm_server_url

    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=build_dataset(args.n_per_env),
        reward_funcs=REWARD_FUNCS,
        args=GRPOConfig(**grpo_kwargs),
        # Dict form: TRL selects the factory per example from the `environment`
        # column and exposes only that env's tools in the episode's schema.
        environment_factory={"guess": SimpleGuessEnv, "agent_tools": AgentToolEnv},
    )
    trainer.train()


if __name__ == "__main__":
    main()
