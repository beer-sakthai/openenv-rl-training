"""Unit tests for evaluate_baselines.py (heuristic baselines & GRPO reward-variance checks)."""

import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluate_baselines import (
    AlwaysStayCatchPolicy,
    EpsilonGreedyCatchPolicy,
    PolicyResult,
    RandomCatchPolicy,
    SmartCatchPolicy,
    _compute_frac_zero_std,
    evaluate_catch_policy,
    evaluate_guess_policy,
)


def test_compute_frac_zero_std_all_uniform():
    """All identical rewards yield frac_zero_std == 1.0 (no advantage)."""
    rewards = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    frac = _compute_frac_zero_std(rewards, group_size=4)
    assert frac == 1.0


def test_compute_frac_zero_std_all_varying():
    """Varying rewards within every group yield frac_zero_std == 0.0 (good advantage)."""
    rewards = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    frac = _compute_frac_zero_std(rewards, group_size=4)
    assert frac == 0.0


def test_compute_frac_zero_std_mixed():
    """One uniform group and one varying group yield frac_zero_std == 0.5."""
    rewards = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    frac = _compute_frac_zero_std(rewards, group_size=4)
    assert frac == 0.5


def test_compute_frac_zero_std_empty_or_small():
    """Empty list or group_size <= 1 defaults to 1.0."""
    assert _compute_frac_zero_std([]) == 1.0
    assert _compute_frac_zero_std([1.0, 0.0], group_size=1) == 1.0


def test_policy_result_win_rate():
    res = PolicyResult(
        env_name="test_env",
        policy_name="test_pol",
        episodes=50,
        wins=25,
        total_reward=25.0,
        avg_steps=5.0,
        frac_reward_zero_std=0.2,
    )
    assert res.win_rate == 0.5

    empty_res = PolicyResult(
        env_name="test_env",
        policy_name="empty",
        episodes=0,
        wins=0,
        total_reward=0.0,
        avg_steps=0.0,
        frac_reward_zero_std=1.0,
    )
    assert empty_res.win_rate == 0.0


def test_catch_policies_action_selection():
    rng = random.Random(123)
    rand_pol = RandomCatchPolicy(rng)
    stay_pol = AlwaysStayCatchPolicy()
    smart_pol = SmartCatchPolicy()
    eps_pol = EpsilonGreedyCatchPolicy(epsilon=0.0, rng=rng)

    legal = [0, 1, 2]
    # Grid where ball is at (0, 0) and paddle is at (9, 2)
    grid = [0.0] * 50
    grid[0] = 1.0  # ball at col 0, row 0
    grid[47] = 1.0  # paddle at col 2, row 9

    assert stay_pol.select_action(grid, legal) == 1
    assert rand_pol.select_action(grid, legal) in legal
    # Smart policy should steer left toward col 0 (action 0: left)
    assert smart_pol.select_action(grid, legal) == 0
    assert eps_pol.select_action(grid, legal) == 0


def test_evaluate_guess_policies():
    """AlwaysZero vs OptimalBisect on SimpleGuessEnv."""
    zero_res = evaluate_guess_policy("always_zero", num_episodes=16)
    assert zero_res.episodes == 16
    assert zero_res.win_rate < 0.2

    opt_res = evaluate_guess_policy("bisect", num_episodes=16)
    assert opt_res.episodes == 16
    assert opt_res.win_rate == 1.0
    assert opt_res.avg_steps < 8.0

    rand_res = evaluate_guess_policy("random", num_episodes=16)
    assert rand_res.episodes == 16

    eps_res = evaluate_guess_policy("epsilon_bisect", num_episodes=16)
    assert eps_res.episodes == 16


def test_evaluate_catch_policy():
    """SmartCatchPolicy wins every episode on simulated Catch."""
    res = evaluate_catch_policy(SmartCatchPolicy(), num_episodes=16)
    assert res.episodes == 16
    assert res.win_rate == 1.0
    assert res.avg_steps == 9.0

    stay_res = evaluate_catch_policy(AlwaysStayCatchPolicy(), num_episodes=16)
    assert stay_res.episodes == 16
    assert stay_res.avg_steps == 9.0
