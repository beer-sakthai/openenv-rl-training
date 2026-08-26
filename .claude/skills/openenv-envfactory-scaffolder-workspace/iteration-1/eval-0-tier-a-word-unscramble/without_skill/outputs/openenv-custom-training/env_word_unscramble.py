"""Tier A: word-unscramble environment (inline plain Python, no server).

The task: the model is shown a scrambled word (e.g. 'ranbcah' for 'branch')
and must submit the unscrambled version via the `submit(word)` tool. An
optional `hint()` tool reveals one correct letter at its correct position.

This is a Tier A environment — the tool methods only compare strings the
policy produces against the hidden target; they never `exec`/`eval`/shell
out on model-produced input. See `env_simple_task.py` for the same shape on
a number-guessing task, and the CLAUDE.md "environment_factory contract"
section for the full ruleset. In particular:

  - `__init__(self)` takes NO arguments (TRL calls `EnvironmentFactory()`).
  - `reset(**kwargs)` receives every dataset column as a keyword arg;
    `target` is scoring-only and never shown to the model. The `scrambled`
    column drives the observation.
  - Every public method other than `reset` becomes a callable tool. Named
    tools (`submit`, `hint`) — not a generic `step(action)`.
  - Google-style docstrings with a full `Args:` block are mandatory; TRL
    reads them via `transformers.get_json_schema`.
  - Episode state lives on `self` (`self.reward`, `self.done`); the reward
    function reads `env.reward` back off the instance after the episode.
  - Raising an exception is how you reject a call — TRL catches it and
    feeds the message back to the model as the tool result.

Binary reward (1.0 solved / 0.0 not), outcome-based. Attempts are capped
so a non-converging rollout terminates cleanly with 0.0 rather than
looping forever.

Run the trainer with:  python train.py --env word_unscramble [--model ...]
"""

import random

from datasets import Dataset

# Attempts capped so a model that never converges still terminates and the
# group can train on a clean 0.0. `hint()` does NOT count against this cap;
# only `submit()` does — matches the intuition that "attempts" are guesses.
MAX_ATTEMPTS = 6


class WordUnscrambleEnv:
    """The class GRPOTrainer instantiates once per generation slot."""

    def __init__(self):
        self._target = ""
        self._scrambled = ""
        self._attempts = 0
        self._revealed: dict[int, str] = {}
        self.reward = 0.0
        self.done = False

    def reset(self, **kwargs) -> str | None:
        # `target` and `scrambled` come from the dataset columns. `target`
        # is used only for scoring — the model never sees it directly.
        self._target = str(kwargs["target"]).lower()
        self._scrambled = str(kwargs["scrambled"]).lower()
        self._attempts = 0
        self._revealed = {}
        self.reward = 0.0
        self.done = False
        # First observation the model sees.
        return (
            f"Scrambled word: '{self._scrambled}' "
            f"({len(self._target)} letters). "
            f"Call submit(word) with the unscrambled word. "
            f"You have {MAX_ATTEMPTS} attempts. "
            f"You may call hint() to reveal one correct letter position; "
            f"hint() does not consume an attempt."
        )

    # -- tools ---------------------------------------------------------------

    def submit(self, word: str) -> str:
        """Submit a candidate unscrambled word.

        Returns 'Correct!' if the word matches the hidden target, otherwise
        feedback about how many letters match at their correct positions
        (0..len(target)). Case-insensitive. Submitting a word of the wrong
        length wastes an attempt and returns a length-mismatch note.

        Args:
            word: your guess for the unscrambled word.
        """
        if self.done:
            raise ValueError("Episode already ended.")
        guess = str(word).strip().lower()
        self._attempts += 1
        if guess == self._target:
            self.reward = 1.0
            self.done = True
            return f"Correct! The word was '{self._target}'."
        if len(guess) != len(self._target):
            remaining = MAX_ATTEMPTS - self._attempts
            if remaining <= 0:
                self.done = True
                return (
                    f"Out of attempts. Wrong length "
                    f"(expected {len(self._target)}, got {len(guess)}). "
                    f"The word was '{self._target}'."
                )
            return (
                f"Wrong length: expected {len(self._target)} letters, "
                f"got {len(guess)}. {remaining} attempt(s) left."
            )
        matches = sum(1 for a, b in zip(guess, self._target) if a == b)
        if self._attempts >= MAX_ATTEMPTS:
            self.done = True
            return (
                f"Out of attempts. '{guess}' matched {matches}/{len(self._target)} "
                f"positions. The word was '{self._target}'."
            )
        remaining = MAX_ATTEMPTS - self._attempts
        return (
            f"Not quite: '{guess}' matches {matches}/{len(self._target)} "
            f"positions. {remaining} attempt(s) left."
        )

    def hint(self) -> str:
        """Reveal one correct letter at its correct position.

        Returns a string like "position 2 is 'a'". Repeated calls reveal
        additional positions until every position has been revealed.
        Does NOT consume a submit attempt. Takes no arguments.
        """
        if self.done:
            raise ValueError("Episode already ended.")
        remaining_positions = [
            i for i in range(len(self._target)) if i not in self._revealed
        ]
        if not remaining_positions:
            return (
                "All positions already revealed: "
                + " ".join(
                    f"pos {i}='{self._revealed[i]}'"
                    for i in sorted(self._revealed)
                )
            )
        # Deterministic pick (leftmost unrevealed) — reproducible rollouts.
        i = remaining_positions[0]
        letter = self._target[i]
        self._revealed[i] = letter
        return f"Hint: position {i} is '{letter}'."


def reward_func(environments, **kwargs) -> list[float]:
    """Binary, outcome-based: 1.0 iff the target was submitted.

    Judges the final state (env.reward), not the path — GRPO can then
    prefer strategies (with/without hints) that actually solve, without
    prescribing any particular one.
    """
    return [env.reward for env in environments]


# A small deterministic word bank. Kept short/common so a 1.5B-tools model
# has a plausible shot at solving some rollouts — GRPO needs reward
# variance within a group to learn anything (see FINDINGS.md).
_WORD_BANK = [
    "branch", "python", "orange", "planet", "silver", "guitar",
    "window", "coffee", "monkey", "purple", "school", "castle",
    "camera", "market", "friend", "letter", "forest", "circle",
    "island", "bottle", "pencil", "garden", "yellow", "winter",
    "summer", "spring", "autumn", "candle", "dragon", "hunter",
    "rocket", "master", "poster", "silver", "temple", "bridge",
]


def _scramble(word: str, rng: random.Random) -> str:
    """Return a permutation of `word` that is guaranteed different from it
    (as long as the word has >= 2 distinct letters)."""
    letters = list(word)
    if len(set(letters)) < 2:
        return word  # nothing to scramble
    for _ in range(32):
        rng.shuffle(letters)
        scrambled = "".join(letters)
        if scrambled != word:
            return scrambled
    # Fallback: rotate by one.
    return word[1:] + word[0]


def build_dataset(n_episodes: int = 64) -> Dataset:
    """One episode per row.

    `target` and `scrambled` are scoring/observation columns forwarded to
    `reset(**kwargs)`; the model sees only `prompt` (which references the
    scrambled word by value, so the environment's first observation and
    the user prompt agree).
    """
    prompts, targets, scrambleds = [], [], []
    # Deterministic RNG so runs are reproducible (mirrors the (i*7)%100
    # trick in env_simple_task.py).
    rng = random.Random(0xC0FFEE)
    for i in range(n_episodes):
        target = _WORD_BANK[i % len(_WORD_BANK)]
        scrambled = _scramble(target, rng)
        prompts.append([
            {
                "role": "user",
                "content": (
                    f"Unscramble the word '{scrambled}'. Call submit(word) with "
                    f"your guess. You have {MAX_ATTEMPTS} attempts. You may call "
                    f"hint() to reveal one correct letter position; hint() does "
                    f"not consume an attempt."
                ),
            }
        ])
        targets.append(target)
        scrambleds.append(scrambled)
    return Dataset.from_dict(
        {"prompt": prompts, "target": targets, "scrambled": scrambleds}
    )


# Reference points for train.py's dispatch table.
ENVIRONMENT_FACTORY = WordUnscrambleEnv
REWARD_FUNCS = reward_func
TRAIN_DATASET = build_dataset
# A SakThai tools model (chat template supports tool calling — required
# by GRPOTrainer(environment_factory=...)). 1.5B fits a Tier A task and
# is smaller than the 7B default `train.py` uses for BrowserGym; override
# with --model or TRAIN_BASE for a fuller run.
DEFAULT_MODEL = "Nanthasit/sakthai-context-1.5b-tools"
