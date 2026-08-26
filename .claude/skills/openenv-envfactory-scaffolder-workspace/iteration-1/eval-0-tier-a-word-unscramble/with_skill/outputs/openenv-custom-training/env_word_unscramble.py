"""Tier A env for word unscramble.

The model is shown a scrambled word (e.g. 'ranbcah') and must submit the
original English word (e.g. 'branch') via `guess(word=...)`. It may spend
`hint()` calls to reveal one correct letter position at a time (leftmost
first), but the final position is never revealed — the model must deduce
at least one letter itself.

Tier A because no tool executes model-produced strings — the tools compare
the guess against the target and mutate scoreboard state. There is nothing
here that would benefit from container isolation; running this in-process
is legitimate.

Landing zone: `openenv-custom-training/env_word_unscramble.py`.
Register in `train.py`'s dispatch table under `--env word_unscramble`.
"""

from __future__ import annotations

import random

from datasets import Dataset

# --- Episode cap ------------------------------------------------------------
# User specified 6 guess attempts. Hints are counted separately (limited by
# word length - 1 so the model must always deduce at least one letter).
# A non-converging rollout terminates with a clean 0.0 after MAX_ATTEMPTS
# guesses, which is what GRPO's within-group ranking needs.
MAX_ATTEMPTS = 6

# --- Word bank --------------------------------------------------------------
# Fixed tuple, all common 6-letter English words with no repeated letters
# (so every position is unambiguously identifiable in a hint). Kept small
# and shared across episodes deliberately — GRPO learns the shape of the
# task from repeated exposure, not from a huge unique-word set.
WORDS: tuple[str, ...] = (
    "branch",
    "garden",
    "planet",
    "silver",
    "purple",
    "monkey",
    "basket",
    "winter",
    "market",
    "forest",
    "bridge",
    "castle",
    "doctor",
    "puzzle",
    "wisdom",
    "orange",
    "pencil",
    "shrimp",
    "quartz",
    "jacket",
)


def _scramble(word: str) -> str:
    """Deterministic scramble seeded by the word itself.

    Same word → same scramble across processes, because `random.Random(str)`
    hashes the string with a stable algorithm (unaffected by
    PYTHONHASHSEED). Retries up to 10 times if the shuffle happens to
    produce the original word.
    """
    rng = random.Random(word)
    letters = list(word)
    for _ in range(10):
        rng.shuffle(letters)
        candidate = "".join(letters)
        if candidate != word:
            return candidate
    return candidate  # last resort — accept even if equal


class WordUnscrambleEnv:
    """The class GRPOTrainer instantiates once per generation in the batch.

    Episode state lives on `self` (one instance per slot; no cross-episode
    bleed). `self.reward` and `self.done` are the two fields the reward
    function reads back.
    """

    def __init__(self):
        # NO arguments — TRL always instantiates as EnvironmentFactory().
        # Config, if we needed any, would come from module-level constants
        # or env vars.
        self._target: str = ""
        self._scrambled: str = ""
        self._attempts: int = 0
        self._revealed: set[int] = set()
        # Init reward/done so a never-stepped rollout returns 0.0, not
        # AttributeError.
        self.reward: float = 0.0
        self.done: bool = False

    def reset(self, **kwargs) -> str:
        """Called once per episode, before any tool call.

        `target` is the hidden true word, passed via a scoring-only dataset
        column the model never sees. The scramble is computed here and
        returned as the model's first observation.
        """
        self._target = str(kwargs["target"]).lower()
        self._scrambled = _scramble(self._target)
        self._attempts = 0
        self._revealed = set()
        self.reward = 0.0
        self.done = False
        return (
            f"Unscramble this word: '{self._scrambled}'. "
            f"Submit your answer with guess(word=...). "
            f"You have {MAX_ATTEMPTS} guesses. "
            "Call hint() to reveal one correct letter position (free, but "
            "the last letter is never revealed)."
        )

    # -- tools ---------------------------------------------------------------
    # Every public method other than reset becomes a callable tool named
    # after the method. Named tools (guess, hint) beat a generic
    # step(action) — the model reads the docstring as the tool description,
    # and a crisp affordance is easier for GRPO to reinforce.

    def guess(self, word: str) -> str:
        """Submit your candidate unscrambled word.

        Compared case-insensitively against the hidden target. A guess whose
        length or letter multiset does not match the scramble is rejected
        with a ValueError (the model sees the message and can retry — the
        rejected call does NOT count against your attempt budget). A valid
        guess that is wrong burns one attempt and returns the count of
        guesses remaining. When you run out of attempts the episode ends
        with reward 0.0.

        Args:
            word: the word you think the scramble unscrambles to.
        """
        if self.done:
            # Rejecting further calls keeps a stray post-terminal tool call
            # from mutating scoring state.
            raise ValueError("Episode already ended.")
        candidate = str(word).strip().lower()
        if len(candidate) != len(self._target):
            # ValueError → the message flows back to the model as the tool
            # result. Actionable feedback beats a silent no-op.
            raise ValueError(
                f"'{candidate}' has {len(candidate)} letters; the target has "
                f"{len(self._target)}. This did not count as a guess."
            )
        if sorted(candidate) != sorted(self._target):
            raise ValueError(
                f"'{candidate}' is not a permutation of '{self._scrambled}'. "
                "This did not count as a guess."
            )
        self._attempts += 1
        if candidate == self._target:
            # Set reward + done together on the terminal step. The reward
            # function reads env.reward off the instance after the episode.
            self.reward = 1.0
            self.done = True
            return f"Correct! The word was '{self._target}'."
        if self._attempts >= MAX_ATTEMPTS:
            self.done = True
            return (
                f"Out of attempts after {MAX_ATTEMPTS} guesses. "
                f"The word was '{self._target}'."
            )
        remaining = MAX_ATTEMPTS - self._attempts
        return (
            f"'{candidate}' is not the word. "
            f"{remaining} guess(es) remaining."
        )

    def hint(self) -> str:
        """Reveal one correct letter of the hidden word (positional).

        Each call fills in the leftmost still-unknown position with the
        true letter that belongs there. Does NOT count against your guess
        budget. The response includes the running underscore pattern so
        you can see progress. Refuses to reveal the last remaining letter
        — you must always deduce at least one position yourself.
        """
        if self.done:
            raise ValueError("Episode already ended.")
        # If revealing another letter would leave zero unknowns, refuse.
        # This is what keeps hint from being a free win: the model always
        # has to reason about at least one position.
        if len(self._revealed) + 1 >= len(self._target):
            raise ValueError(
                "Cannot reveal the last letter — you must deduce it. "
                "Submit your answer with guess(word=...)."
            )
        for i, ch in enumerate(self._target):
            if i not in self._revealed:
                self._revealed.add(i)
                pattern = "".join(
                    c if j in self._revealed else "_"
                    for j, c in enumerate(self._target)
                )
                return (
                    f"Position {i} is '{ch}'. Pattern so far: {pattern}"
                )
        # All positions already revealed (shouldn't reach here given the
        # guard above, but keep the safety net).
        raise ValueError("No positions left to reveal.")


def reward_func(environments, **kwargs) -> list[float]:
    """Binary, outcome-based: 1.0 iff the model landed on the target word.

    Signature is (environments, **kwargs) -> list[float] per the TRL
    contract. Reads env.reward off each instance — the env sets it 1.0 on
    a correct guess and leaves it at 0.0 otherwise, including on timeouts.
    """
    return [env.reward for env in environments]


def build_dataset(n_episodes: int = 64) -> Dataset:
    """One episode per row.

    `target` is a scoring-only column: it is passed to reset(**kwargs) but
    never shown to the model. The model sees only the `prompt` (a
    conversational list of {role, content} dicts) plus the observation
    reset() returns, which contains the scrambled string.

    Deterministic selection via `(i * 7) % len(WORDS)` — same idea as the
    other envs in this repo, keeps runs reproducible.
    """
    prompts: list[list[dict]] = []
    targets: list[str] = []
    for i in range(n_episodes):
        target = WORDS[(i * 7) % len(WORDS)]
        # prompt MUST be a list of {role, content} dicts, not a bare
        # string — TRL's tool-calling GRPO does prompt[-1]['content'] and
        # a string crashes with TypeError on the first batch.
        prompts.append([
            {
                "role": "user",
                "content": (
                    "You are given a scrambled English word. Unscramble it "
                    "and submit the original word with the guess(word=...) "
                    f"tool. You have {MAX_ATTEMPTS} guess attempts. You may "
                    "call hint() to reveal one correct letter position — "
                    "hints do not count against your guess budget, but the "
                    "final letter is never revealed. Only real English "
                    "words are valid answers."
                ),
            }
        ])
        targets.append(target)
    return Dataset.from_dict({"prompt": prompts, "target": targets})


# --- Module-level exports ---------------------------------------------------
# train.py's dispatch table looks these names up by convention.
ENVIRONMENT_FACTORY = WordUnscrambleEnv
REWARD_FUNCS = reward_func
TRAIN_DATASET = build_dataset

# Default base — one of the SakThai tools models per the user's request.
# 1.5b-tools is a reasonable size for this task (short output, small
# vocabulary of letter sequences). Do NOT default to -merged bases; they
# ship no chat template with tool-calling and GRPOTrainer raises at
# construction.
DEFAULT_MODEL = "Nanthasit/sakthai-context-1.5b-tools"
