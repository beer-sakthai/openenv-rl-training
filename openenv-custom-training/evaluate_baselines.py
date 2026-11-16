#!/usr/bin/env python3
# /// script
# dependencies = ["datasets", "pydantic"]
# ///
"""Pre-training baseline policy competition and GRPO reward-variance sanity check.

Adapted from the OpenEnv `openenv_using_environments.ipynb` and
`openenv_introduction_quickstart.ipynb` notebooks.

Per `CLAUDE.md` ("Before spending GPU time on a new environment, drive a few
episodes by hand and confirm a capable model scores above random") and
`FINDINGS.md` ("GRPO can only reinforce successes the model samples during
rollouts; zero reward variance -> zero advantage -> zero gradient"):

This script pits baseline policies (`AlwaysWorstPolicy`, `RandomPolicy`,
`EpsilonGreedyPolicy`, `SmartHeuristicPolicy`) across:
  1. `SimpleGuessEnv` (Tier A inline number-guessing)
  2. `SimulatedCatchEnv` (OpenSpiel 10x5 Catch grid from `openenv_using_environments.ipynb`)

It prints a leaderboard with win rates, average steps, and the simulated GRPO
`frac_reward_zero_std` across groups of `G=4` rollouts to verify that the
environment yields non-zero gradient signal before launching a GPU run.

Run on CPU:
    python3 openenv-custom-training/evaluate_baselines.py
"""

import argparse
from dataclasses import dataclass
import math
import os
import random
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_simple_task import MAX_ATTEMPTS, SimpleGuessEnv  # noqa: E402


@dataclass
class PolicyResult:
    """Aggregated evaluation metrics for a single policy on an environment."""

    env_name: str
    policy_name: str
    episodes: int
    wins: int
    total_reward: float
    avg_steps: float
    frac_reward_zero_std: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.episodes if self.episodes else 0.0


def _compute_frac_zero_std(rewards: List[float], group_size: int = 4) -> float:
    """Compute the fraction of GRPO generation groups with zero reward variance."""
    if not rewards or group_size <= 1:
        return 1.0
    zero_groups = 0
    total_groups = 0
    for i in range(0, len(rewards) - group_size + 1, group_size):
        group = rewards[i : i + group_size]
        mean = sum(group) / len(group)
        var = sum((r - mean) ** 2 for r in group) / len(group)
        if math.sqrt(var) < 1e-8:
            zero_groups += 1
        total_groups += 1
    return zero_groups / total_groups if total_groups else 1.0


# ============================================================================
# 1. OpenSpiel Catch Local Simulation (from openenv_using_environments.ipynb)
# ============================================================================

GRID_HEIGHT = 10
GRID_WIDTH = 5


class RandomCatchPolicy:
    name = "RandomPolicy"

    def __init__(self, rng: random.Random):
        self._rng = rng

    def select_action(self, info_state: List[float], legal_actions: List[int]) -> int:
        return self._rng.choice(legal_actions)


class AlwaysStayCatchPolicy:
    name = "AlwaysStayPolicy"

    def select_action(self, info_state: List[float], legal_actions: List[int]) -> int:
        return 1


class SmartCatchPolicy:
    """Optimal heuristic policy from `openenv_using_environments.ipynb`.
    Reads the full 50-element (10x5) grid state, locates the ball and paddle
    columns (paddle is in row 9, indices 45..49), and steps toward the ball.
    """

    name = "SmartCatchPolicy"

    def select_action(self, info_state: List[float], legal_actions: List[int]) -> int:
        ball_col = None
        paddle_col = None
        for idx, val in enumerate(info_state):
            if val > 0.5:
                row = idx // GRID_WIDTH
                col = idx % GRID_WIDTH
                if row < GRID_HEIGHT - 1:
                    ball_col = col
                else:
                    paddle_col = col
        if ball_col is None or paddle_col is None:
            return 1
        if ball_col < paddle_col and 0 in legal_actions:
            return 0
        if ball_col > paddle_col and 2 in legal_actions:
            return 2
        return 1


class EpsilonGreedyCatchPolicy:
    name = "EpsilonGreedy(0.3)"

    def __init__(self, epsilon: float, rng: random.Random):
        self._epsilon = epsilon
        self._rng = rng
        self._smart = SmartCatchPolicy()

    def select_action(self, info_state: List[float], legal_actions: List[int]) -> int:
        if self._rng.random() < self._epsilon:
            return self._rng.choice(legal_actions)
        return self._smart.select_action(info_state, legal_actions)


def evaluate_catch_policy(policy, num_episodes: int = 64, seed: int = 42) -> PolicyResult:
    rng = random.Random(seed)
    rewards: List[float] = []
    steps_list: List[int] = []
    wins = 0

    for _ in range(num_episodes):
        ball_col = rng.randint(0, GRID_WIDTH - 1)
        paddle_col = GRID_WIDTH // 2
        legal_actions = [0, 1, 2]

        for step in range(GRID_HEIGHT - 1):
            info_state = [0.0] * (GRID_HEIGHT * GRID_WIDTH)
            info_state[step * GRID_WIDTH + ball_col] = 1.0
            info_state[(GRID_HEIGHT - 1) * GRID_WIDTH + paddle_col] = 1.0

            action_id = policy.select_action(info_state, legal_actions)
            if action_id == 0:
                paddle_col = max(0, paddle_col - 1)
            elif action_id == 2:
                paddle_col = min(GRID_WIDTH - 1, paddle_col + 1)

        reward = 1.0 if paddle_col == ball_col else 0.0
        rewards.append(reward)
        steps_list.append(GRID_HEIGHT - 1)
        if reward > 0:
            wins += 1

    return PolicyResult(
        env_name="OpenSpiel:Catch(50)",
        policy_name=policy.name,
        episodes=num_episodes,
        wins=wins,
        total_reward=sum(rewards),
        avg_steps=sum(steps_list) / len(steps_list),
        frac_reward_zero_std=_compute_frac_zero_std(rewards),
    )


# ============================================================================
# 2. Tier A SimpleGuessEnv Policies
# ============================================================================


def evaluate_guess_policy(
    mode: str, num_episodes: int = 64, seed: int = 42
) -> PolicyResult:
    rng = random.Random(seed)
    env = SimpleGuessEnv()
    rewards: List[float] = []
    steps_list: List[int] = []
    wins = 0

    for i in range(num_episodes):
        target = (i * 7) % 100
        env.reset(target=target)
        low, high = 0, 99
        steps = 0

        while not env.done and steps < MAX_ATTEMPTS:
            if mode == "always_zero":
                guess_val = 0
            elif mode == "random":
                guess_val = rng.randint(0, 99)
            elif mode == "epsilon_bisect":
                if rng.random() < 0.35:
                    guess_val = rng.randint(low, high)
                else:
                    guess_val = (low + high) // 2
            elif mode == "bisect":
                guess_val = (low + high) // 2
            else:
                raise ValueError(f"Unknown mode: {mode}")

            feedback = env.guess(guess_val)
            steps += 1
            if feedback == "higher":
                low = max(low, guess_val + 1)
            elif feedback == "lower":
                high = min(high, guess_val - 1)

        rewards.append(env.reward)
        steps_list.append(steps)
        if env.reward > 0:
            wins += 1

    names = {
        "always_zero": "AlwaysZeroPolicy",
        "random": "RandomGuessPolicy",
        "epsilon_bisect": "NoisyBisect(0.35)",
        "bisect": "OptimalBisectPolicy",
    }
    return PolicyResult(
        env_name="TierA:SimpleGuessEnv",
        policy_name=names[mode],
        episodes=num_episodes,
        wins=wins,
        total_reward=sum(rewards),
        avg_steps=sum(steps_list) / len(steps_list),
        frac_reward_zero_std=_compute_frac_zero_std(rewards),
    )


def run_all_competitions(num_episodes: int = 64) -> List[PolicyResult]:
    rng = random.Random(42)
    catch_policies = [
        AlwaysStayCatchPolicy(),
        RandomCatchPolicy(rng),
        EpsilonGreedyCatchPolicy(0.3, rng),
        SmartCatchPolicy(),
    ]
    results: List[PolicyResult] = []
    for p in catch_policies:
        results.append(evaluate_catch_policy(p, num_episodes=num_episodes))
    for mode in ("always_zero", "random", "epsilon_bisect", "bisect"):
        results.append(evaluate_guess_policy(mode, num_episodes=num_episodes))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run baseline policy competition and GRPO reward-variance checks."
    )
    parser.add_argument("--episodes", type=int, default=64)
    args = parser.parse_args()

    results = run_all_competitions(num_episodes=args.episodes)

    print("=" * 84)
    print("   OPENENV BASELINE POLICY COMPETITION & GRPO REWARD-VARIANCE CHECK")
    print("=" * 84)
    print(
        f"{'Environment':<22}{'Policy':<22}{'Win Rate':<12}{'Avg Steps':<12}"
        f"{'frac_zero_std':<15}"
    )
    print("-" * 84)
    for r in results:
        print(
            f"{r.env_name:<22}{r.policy_name:<22}{r.win_rate * 100:>6.1f}%{'':<5}"
            f"{r.avg_steps:>6.1f}{'':<6}{r.frac_reward_zero_std:>8.2f}"
        )
    print("-" * 84)
    print(
        "Note: GRPO requires 0 < win_rate < 100% (so `frac_reward_zero_std` < 1.0)\n"
        "for non-zero advantage and gradient signal within generation groups."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
