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

BrowserGym Space: https://huggingface.co/spaces/openenv/browsergym_env
BrowserGym URL:   https://openenv-browsergym-env.hf.space
"""

import argparse
import os

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

# The upstream OpenEnv catalog Space. Verified 2026-09-16: `/health` returns
# {"status":"healthy"} and `/openapi.json` lists the OpenEnv surface
# (/reset, /step, /state, /schema, /metadata).
#
# This used to default to `Nanthasit/browsergym-env`, which NO LONGER EXISTS —
# `hf stat` reports it missing and both `/` and `/health` on
# nanthasit-browsergym-env.hf.space return 404 (checked twice on 2026-09-16).
# If you redeploy it from `browsergym-space/`, point `--browsergym-url` (or
# TRAIN_BROWSERGYM_URL) back at your own Space.
BROWSERGYM_SPACE_URL = "https://openenv-browsergym-env.hf.space"


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# Episodes are capped so a policy that never solves the task still terminates
# and the group trains on a clean 0.0 rather than looping (repo convention;
# see env_simple_task.MAX_ATTEMPTS and sandbox_env.STEP_LIMIT).
BROWSERGYM_STEP_LIMIT = 30

BROWSERGYM_PROMPT = (
    "You are a web navigation agent. Complete the task shown in the browser. "
    "Call the run_action(action) tool with one BrowserGym action per call — "
    "click(), fill(), goto(), press(), or scroll(). Observe the page carefully "
    "and act step by step until the task is done."
)


def _obs_text(observation) -> str:
    """Best-effort render of a BrowserGym observation as the model's tool result.

    NOT VERIFIED against a live `browsergym_env` install (see
    _BrowserGymTaskEnv's docstring) — the field order below is a preference
    list, not a checked schema.
    """
    for field in ("text", "observation", "result", "page", "axtree", "dom"):
        value = getattr(observation, field, None)
        if isinstance(value, str) and value:
            return value
    return str(observation)


class _BrowserGymTaskEnv:
    """TRL-facing wrapper around one BrowserGym OpenEnv session.

    Mirrors the Tier B shape in `agent_tools/wrapper.py`: hold a client,
    translate a named tool call into `client.step(...)`, and keep episode state
    on `self` so the reward function can read `env.reward` back off the
    instance after the episode. The server owns the verdict; this forwards it.

    Instances are built by the closure `_make_browsergym_factory` returns, so
    the per-run task name and Space URL arrive without `__init__` taking
    arguments — TRL calls the factory with none.

    **Written, not run.** This wrapper has not been executed against any
    BrowserGym server (no GPU in the checkout where it was written).

    An earlier revision of this docstring claimed the Space at
    `Nanthasit/browsergym-env` was live. It is not, and was not then: that
    Space does not exist, and both `/` and `/health` on
    nanthasit-browsergym-env.hf.space return 404. The claim was carried over
    from README.md rather than checked. The default now points at the upstream
    catalog Space, whose OpenEnv surface was verified on 2026-09-16 (see
    BROWSERGYM_SPACE_URL above).

    The client surface used here — `.reset()` / `.step()` returning a
    `StepResult` with `.observation`, `.reward`, `.done` — is the OpenEnv
    `EnvClient` contract that `agent_tools/client.py` documents, and
    `/openapi.json` on the upstream Space confirms the matching `/reset` and
    `/step` routes exist. The BrowserGym *action* type remains the unverified
    part: `BrowserGymAction(action=...)` is assumed below, as is the
    observation text field. Diff both against the installed `browsergym_env`
    package before spending GPU time.
    """

    def __init__(self, task_name: str, space_url: str):
        self._task_name = task_name
        self._space_url = space_url
        self._client = None
        self._steps = 0
        self.reward = 0.0
        self.done = False

    def _open(self):
        try:
            from browsergym_env import BrowserGymEnv
        except ImportError:
            raise ImportError(
                "browsergym_env not installed. "
                "Run: pip install git+https://github.com/huggingface/OpenEnv.git"
                "#subdirectory=envs/browsergym_env"
            )
        return BrowserGymEnv(
            base_url=self._space_url,
            environment={
                "BROWSERGYM_BENCHMARK": "miniwob",
                "BROWSERGYM_TASK_NAME": self._task_name,
                "BROWSERGYM_HEADLESS": "true",
            },
        )

    def reset(self, **kwargs) -> str | None:
        self._steps = 0
        self.reward = 0.0
        self.done = False
        self._client = self._open()
        result = self._client.reset()
        return _obs_text(getattr(result, "observation", result))

    # -- tools ---------------------------------------------------------------

    def run_action(self, action: str) -> str:
        """Perform one BrowserGym action in the browser and see the new page.

        One action per call. Use the BrowserGym action syntax — for example
        `click('a12')`, `fill('a5', 'hello')`, `press('a5', 'Enter')`,
        `goto('http://…')`, or `scroll(0, 200)`. Element ids come from the
        page description in the previous observation. Keep acting until the
        page shows the task is complete.

        Args:
            action: a single BrowserGym action call, e.g. "click('a12')".
        """
        if self.done:
            raise ValueError("Episode already ended.")
        from browsergym_env import BrowserGymAction

        result = self._client.step(BrowserGymAction(action=action))
        self._steps += 1
        observation = getattr(result, "observation", result)
        # Reward/done are mirrored at the StepResult top level and on the
        # observation; prefer the top level, same as agent_tools/client.py.
        reward = getattr(result, "reward", None)
        if reward is None:
            reward = getattr(observation, "reward", 0.0)
        self.reward = float(reward or 0.0)
        self.done = bool(
            getattr(result, "done", None) or getattr(observation, "done", False)
        )
        text = _obs_text(observation)
        if not self.done and self._steps >= BROWSERGYM_STEP_LIMIT:
            self.done = True
            text = f"{text}\n\nSTEP LIMIT REACHED ({BROWSERGYM_STEP_LIMIT})."
        return text


def _make_browsergym_factory(task_name: str, space_url: str):
    """Build a GRPOTrainer-compatible environment factory for BrowserGym.

    TRL instantiates one environment per generation slot by calling this with
    no arguments, so the task name and Space URL ride in on the closure.
    """

    def factory():
        return _BrowserGymTaskEnv(task_name, space_url)

    return factory


def _browsergym_reward(environments, **kwargs):
    """Forward the BrowserGym server's episode reward back to GRPO.

    Signature is the repo-wide contract — `reward_func(environments, **kwargs)`
    reading `env.reward` off the instance after the episode — not
    `(completions, ...)`. Outcome-based: the server judges, this forwards.
    """
    return [float(getattr(env, "reward", 0.0) or 0.0) for env in environments]


def _browsergym_dataset(n_episodes: int):
    """Minimal prompt dataset for BrowserGym web navigation tasks.

    `prompt` is conversational (a list of {"role", "content"} dicts): TRL's
    tool-calling GRPO does `prompt[-1]["content"]`, and a bare string raises
    `TypeError: string indices must be integers`.
    """
    prompts = [
        {"prompt": [{"role": "user", "content": BROWSERGYM_PROMPT}]}
        for _ in range(n_episodes)
    ]
    return Dataset.from_list(prompts)


BROWSERGYM_DEFAULT_MODEL = "Nanthasit/sakthai-context-7b-tools"
BROWSERGYM_CODER_MODEL = "Nanthasit/sakthai-coder-browser"


def _select(args: argparse.Namespace):
    """Return (factory, reward_funcs, dataset_builder, default_model)."""
    if args.env == "simple":
        from env_simple_task import (DEFAULT_MODEL, ENVIRONMENT_FACTORY,
                                     REWARD_FUNCS, TRAIN_DATASET)
    elif args.env == "agent_tools":
        from agent_tools.wrapper import (DEFAULT_MODEL, ENVIRONMENT_FACTORY,
                                         REWARD_FUNCS, TRAIN_DATASET)
    elif args.env == "mcp":
        from env_mcp_task import (DEFAULT_MODEL, ENVIRONMENT_FACTORY,
                                  REWARD_FUNCS, TRAIN_DATASET)
    elif args.env == "browsergym":
        task = getattr(args, "browsergym_task", "click-test")
        url = getattr(args, "browsergym_url", BROWSERGYM_SPACE_URL)
        ENVIRONMENT_FACTORY = _make_browsergym_factory(task, url)
        REWARD_FUNCS = [_browsergym_reward]
        TRAIN_DATASET = _browsergym_dataset
        browser_choice = getattr(args, "browsergym_model", "default")
        if browser_choice == "coder":
            DEFAULT_MODEL = BROWSERGYM_CODER_MODEL
        else:
            DEFAULT_MODEL = BROWSERGYM_DEFAULT_MODEL
    else:
        raise SystemExit(
            f"unknown --env {args.env!r}; choose 'simple', 'agent_tools', 'mcp', or 'browsergym'"
        )
    return ENVIRONMENT_FACTORY, REWARD_FUNCS, TRAIN_DATASET, DEFAULT_MODEL


def parse_args():
    env_default = os.environ.get("TRAIN_ENV")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        choices=["simple", "agent_tools", "mcp", "browsergym"],
        default=env_default,
        required=(env_default is None),
    )
    parser.add_argument(
        "--browsergym-task",
        default=os.environ.get("TRAIN_BROWSERGYM_TASK", "click-test"),
        help="MiniWoB task name (default: click-test). e.g. click-button, email-inbox",
    )
    parser.add_argument(
        "--browsergym-url",
        default=os.environ.get("TRAIN_BROWSERGYM_URL", BROWSERGYM_SPACE_URL),
        help=f"BrowserGym Space URL (default: {BROWSERGYM_SPACE_URL})",
    )
    parser.add_argument(
        "--browsergym-model",
        choices=["default", "coder"],
        default=os.environ.get("TRAIN_BROWSERGYM_MODEL", "default"),
        help="Model choice for browsergym: 'default' (7B tools) or 'coder' (1.5B sakthai-coder-browser). Env: TRAIN_BROWSERGYM_MODEL",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("TRAIN_BASE"),
        help="defaults to Nanthasit/sakthai-context-7b-tools (or task default). Env: TRAIN_BASE",
    )
    parser.add_argument(
        "--vllm-mode",
        choices=["colocate", "server"],
        default=os.environ.get("TRAIN_VLLM_MODE", "colocate"),
    )
    parser.add_argument(
        "--vllm-server-host", default=os.environ.get("TRAIN_VLLM_HOST", "localhost")
    )
    parser.add_argument(
        "--vllm-server-port",
        type=int,
        default=int(os.environ.get("TRAIN_VLLM_PORT", "8000")),
    )
    # Caps tokens across the WHOLE multi-turn episode (generations + tool
    # results summed), not one turn - raise if episodes truncate mid-task.
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=int(os.environ.get("TRAIN_MAX_COMPLETION", "1024")),
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=int(os.environ.get("TRAIN_NUM_GENERATIONS", "4")),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=int(os.environ.get("TRAIN_GRAD_ACCUM", "64")),
    )
    parser.add_argument(
        "--n-episodes", type=int, default=int(os.environ.get("TRAIN_EPISODES", "64"))
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=_bool_env("TRAIN_ENABLE_THINKING", False),
        help="flip on for harder tasks; costs more tokens/turn. "
        "Only Qwen3-template bases act on this; Qwen2 templates ignore it.",
    )
    parser.add_argument(
        "--push-to-hub",
        default=os.environ.get("TRAIN_PUSH_TO"),
        help="repo id to push the trained model to (optional). Env: TRAIN_PUSH_TO",
    )
    return parser.parse_args()


def main():
    args = parse_args()

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
