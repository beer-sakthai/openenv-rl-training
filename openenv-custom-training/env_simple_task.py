"""Tier A: a custom environment implemented as inline plain-Python logic.

No OpenEnv server, no Docker, no base_url. This is legitimate — TRL's
`environment_factory` contract only requires an object with `reset()` and
public tool methods; nothing about it requires OpenEnv specifically. Use this
shape when the task logic is safe plain Python and never executes
untrusted/arbitrary code. If a tool method would shell out to run whatever
string the model produced, use the Tier B sandboxed server in `agent_tools/`
instead — GRPO exploration will make the policy try weird actions, and you do
not want that running inside your training process.

The task: a number-guessing game (guess the hidden integer 0..99, getting
"higher"/"lower" feedback). It demonstrates every rule that matters:

  - `__init__(self)` takes NO arguments — TRL instantiates one per
    generation slot: `EnvironmentFactory()`, not `EnvironmentFactory(cfg)`.
  - `reset(**kwargs)` receives every dataset column as a keyword argument.
    Here the hidden `target` is passed in a column the model never sees; the
    environment uses it only to score.
  - every public method other than `reset` becomes a callable tool, named
    after the method, with a schema auto-generated from its type hints and
    docstring. The docstring IS the interface the model reads — write it
    like any tool description. Note the named `guess(number: int)` /
    `log_reasoning(note: str)` tools, not a single generic `step(action)`.
  - episode state lives on `self`; the reward function reads it back off the
    instance after the episode ends.
  - raising an exception ends or rejects: TRL catches it and feeds the
    message back to the model as the tool result. Out-of-range guesses are
    rejected that way; a wrong-but-valid guess is normal feedback.

Binary reward (1.0 solved / 0.0 not), judged on final state only.

Run the trainer with:  python train.py --env simple [--model ...]
"""

from datasets import Dataset

# Episodes capped so a model that never converges still terminates, and the
# group can train on a clean 0.0 rather than an infinite loop.
MAX_ATTEMPTS = 10


class SimpleGuessEnv:
    """The class GRPOTrainer instantiates once per generation in the batch."""

    def __init__(self):
        self._target = -1
        self._attempts = 0
        self.reward = 0.0
        self.done = False

    def reset(self, **kwargs) -> str | None:
        # `target` comes from the dataset column — used for scoring only, the
        # model never sees it. (This is also the routing hook for multi-task
        # classes; see multi_env.py for routing across whole environments.)
        self._target = int(kwargs["target"])
        self._attempts = 0
        self.reward = 0.0
        self.done = False
        # The first observation the model sees. Returning a string is the
        # mechanism for giving the model an initial prompt-free hint.
        return "The number is between 0 and 99. Guess it."

    # -- tools ---------------------------------------------------------------

    def guess(self, number: int) -> str:
        """Submit an integer guess and get back feedback.

        Returns 'higher' or 'lower' to steer you toward the hidden number, or
        'Correct!' when you find it. Raising an invalid guess (outside 0..99)
        wastes a turn. Keep guessing until you get 'Correct!' — do not stop
        after a single wrong guess.

        Args:
            number: the integer you think the hidden number is (0..99).
        """
        if not (0 <= number <= 99):
            raise ValueError(f"{number} is out of range — guess an integer between 0 and 99.")
        self._attempts += 1
        if number == self._target:
            self.reward = 1.0
            self.done = True
            return f"Correct! The number was {self._target}."
        if self._attempts >= MAX_ATTEMPTS:
            # End the episode so a non-converging run terminates; a clean 0.0
            # is better training signal than an unbounded loop.
            self.done = True
            return f"Out of attempts. The number was {self._target}."
        return "higher" if number < self._target else "lower"

    def log_reasoning(self, note: str) -> str:
        """Record a note about your current hypothesis. No effect on scoring.

        Args:
            note: any text describing your reasoning or next plan.
        """
        if self.done:
            raise ValueError("Episode already ended.")
        return f"Noted: {note}"


def reward_func(environments, **kwargs) -> list[float]:
    """Binary, outcome-based: 1.0 iff the episode was solved.

    Judges the final state (env.reward), not the path — the model should be
    free to find any bisection strategy. See README 'Reward philosophy'.
    """
    return [env.reward for env in environments]


def build_dataset(n_episodes: int = 64) -> Dataset:
    """One episode per row. `target` is a scoring-only column: the model sees
    only the `prompt`; `reset(**kwargs)` receives `target` for scoring."""
    prompts = []
    targets = []
    for i in range(n_episodes):
        # Deterministic spread rather than randomness keeps runs reproducible.
        target = (i * 7) % 100
        prompts.append([
            {
                "role": "user",
                "content": (
                    "I'm thinking of a number between 0 and 99. Use the guess(number) "
                    "tool to find it, using the higher/lower feedback. You have "
                    f"{MAX_ATTEMPTS} attempts. You may use log_reasoning(note) to "
                    "think out loud — it is free."
                ),
            }
        ])
        targets.append(target)
    return Dataset.from_dict({"prompt": prompts, "target": targets})


# Reference point for train.py's dispatch table.
ENVIRONMENT_FACTORY = SimpleGuessEnv
REWARD_FUNCS = reward_func
TRAIN_DATASET = build_dataset
DEFAULT_MODEL = "Nanthasit/sakthai-context-1.5b-merged"
