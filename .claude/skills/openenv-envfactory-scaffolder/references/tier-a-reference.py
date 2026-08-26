"""Tier A reference — a full, working environment_factory module.

COPY-ADAPT this file. Every line here is here for a reason; the comments
say which reason. If you delete a comment while adapting, at least keep
the shape.

Task: a number-guessing game (guess the hidden integer 0..99, getting
"higher"/"lower" feedback). Deliberately trivial, because the point is
the SHAPE, not the task — the same shape carries anything from
sentiment classification to arithmetic to constrained decoding to a
turn-based game against the environment.

Landing zone: `openenv-custom-training/env_<yourname>.py`
Register in `train.py`'s dispatch table under `--env <yourname>`.
"""

from datasets import Dataset

# --- Episode cap ------------------------------------------------------------
# Every environment MUST cap its episode length. GRPO ranks within a group;
# one unbounded rollout wastes the whole group. Set this high enough that a
# capable model reliably converges, low enough that a bad rollout terminates
# with a clean 0.0.
MAX_ATTEMPTS = 10


class SimpleGuessEnv:
    """The class GRPOTrainer instantiates once per generation in the batch.

    One instance per generation slot means episode state is fine to keep on
    self — there is no cross-episode bleed. What is NOT fine is stashing
    state in a class attribute or module-level dict: slot-to-slot bleed
    silently breaks training.
    """

    def __init__(self):
        # NO arguments. TRL instantiates as EnvironmentFactory(); passing
        # config here breaks the contract. Config comes from module-level
        # constants or env vars (see agent_tools/wrapper.py for env var use).
        self._target = -1
        self._attempts = 0
        # reward and done are the two fields the reward function reads back.
        # They MUST be initialized here so a never-stepped rollout returns 0.0
        # (which is a real training signal), not AttributeError.
        self.reward = 0.0
        self.done = False

    def reset(self, **kwargs) -> str | None:
        """Called once per episode, before any tool call.

        kwargs contains every dataset column EXCEPT the routing control
        column in dict-form multi-env factories. Here `target` is the
        hidden answer, passed in the dataset as a scoring-only column
        the model never sees.

        Return a string to give the model its first observation, or None
        to skip. Returning a string is the mechanism for a prompt-free
        initial hint.
        """
        self._target = int(kwargs["target"])
        self._attempts = 0
        self.reward = 0.0
        self.done = False
        return "The number is between 0 and 99. Guess it."

    # -- tools ---------------------------------------------------------------
    # Every public method other than reset becomes a callable tool, named
    # after the method. Prefer NAMED tools (guess, log_reasoning) over a
    # single generic step(action). The model reads each docstring as the
    # tool description — write it like a tool description, not a code
    # comment. Google-style docstring with a complete Args: block is
    # MANDATORY. A one-line docstring raises DocstringParsingException at
    # trainer construction, before any step runs.

    def guess(self, number: int) -> str:
        """Submit an integer guess and get back feedback.

        Returns 'higher' or 'lower' to steer you toward the hidden number,
        or 'Correct!' when you find it. Raising an invalid guess (outside
        0..99) wastes a turn. Keep guessing until you get 'Correct!' — do
        not stop after a single wrong guess.

        Args:
            number: the integer you think the hidden number is (0..99).
        """
        # Raising an exception REJECTS the call — TRL catches it and feeds
        # the message back to the model as the tool result. The model can
        # act on that message; a silent no-op it cannot.
        if not (0 <= number <= 99):
            raise ValueError(
                f"{number} is out of range — guess an integer between 0 and 99."
            )
        self._attempts += 1

        if number == self._target:
            # Set reward + done together on the terminal step. The reward
            # function reads env.reward after the episode ends.
            self.reward = 1.0
            self.done = True
            return f"Correct! The number was {self._target}."

        if self._attempts >= MAX_ATTEMPTS:
            # End the episode so a non-converging run terminates. A clean
            # 0.0 is better training signal than an infinite loop.
            self.done = True
            return f"Out of attempts. The number was {self._target}."

        return "higher" if number < self._target else "lower"

    def log_reasoning(self, note: str) -> str:
        """Record a note about your current hypothesis. No effect on scoring.

        Args:
            note: any text describing your reasoning or next plan.
        """
        # A "thinking" tool with no scoring effect is useful when the model
        # benefits from an explicit step for reasoning between tool calls.
        # Guarding on self.done keeps the model honest — no post-hoc log
        # entries after the episode ended.
        if self.done:
            raise ValueError("Episode already ended.")
        return f"Noted: {note}"


def reward_func(environments, **kwargs) -> list[float]:
    """Binary, outcome-based: 1.0 iff the episode was solved.

    Signature must be (environments, **kwargs) -> list[float]. The trainer
    passes the full batch of env instances; the reward function reads
    reward off each one.

    Judge on the FINAL state (env.reward), not the path — GRPO ranks
    within a group, so only ordering matters. An outcome-only reward lets
    the model find any strategy that works; a path-scored reward locks it
    into whatever plan you had in mind and starves exploration.
    """
    return [env.reward for env in environments]


def build_dataset(n_episodes: int = 64) -> Dataset:
    """One episode per row. `target` is a scoring-only column: the model sees
    only `prompt`; reset(**kwargs) receives `target`.

    Deterministic construction — do NOT use random.randint. Reproducible
    runs matter for the eval-then-train loop.
    """
    prompts = []
    targets = []
    for i in range(n_episodes):
        # (i * 7) % 100 spreads targets evenly with a small prime step so
        # consecutive episodes have unrelated answers. Same idea as the
        # other envs in this repo.
        target = (i * 7) % 100
        # prompt MUST be a list of {role, content} dicts, not a plain
        # string. TRL's tool-calling GRPO does prompt[-1]["content"] and a
        # bare string crashes with TypeError on the first batch.
        prompts.append([
            {
                "role": "user",
                "content": (
                    "I'm thinking of a number between 0 and 99. Use the "
                    "guess(number) tool to find it, using the higher/lower "
                    f"feedback. You have {MAX_ATTEMPTS} attempts. You may "
                    "use log_reasoning(note) to think out loud — it is free."
                ),
            }
        ])
        targets.append(target)
    return Dataset.from_dict({"prompt": prompts, "target": targets})


# --- Module-level exports ---------------------------------------------------
# train.py's dispatch table looks these names up by convention. Keep them
# spelled exactly like this.
ENVIRONMENT_FACTORY = SimpleGuessEnv
REWARD_FUNCS = reward_func
TRAIN_DATASET = build_dataset

# Default base model — the scaffolder should set this to whichever base is
# appropriate for the difficulty of the task:
#   - trivial task, 0.5-1.5B is enough: "Nanthasit/sakthai-context-1.5b-tools"
#   - realistic tool-calling: "Nanthasit/sakthai-context-7b-tools"
# Do NOT default to sakthai-context-{0.5,1.5,7}b-merged — they ship no
# chat template with tool-calling support and GRPOTrainer will raise.
DEFAULT_MODEL = "Nanthasit/sakthai-context-1.5b-tools"
