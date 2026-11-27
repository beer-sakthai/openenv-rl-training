#!/usr/bin/env python3
"""verify_grpo_contract.py — CPU-only contract check for the custom environments.

Verifies, against the repo's real modules (not against a mock defined in this
file):

1. `SimpleGuessEnv` honours the `environment_factory` contract — no-arg
   `__init__`, `reset(**kwargs)` taking scoring-only dataset columns, named
   tool methods with Google-style `Args:` blocks, episode state on `self`,
   an exception to reject an invalid call, and a capped episode.
2. A solvable episode driven by hand reaches `reward == 1.0` and an
   unsolvable one terminates at `MAX_ATTEMPTS` with `0.0` — the "drive a few
   episodes by hand before spending GPU time" check from CLAUDE.md.
3. `reward_func` returns one float per environment, read off `env.reward`.
4. `GRPOConfig` accepts `vllm_server_host`/`vllm_server_port` (the fields that
   actually exist) and rejects `vllm_server_url` (which crashes at
   construction).

Sections 1-3 need `datasets` (env_simple_task imports it at module level) and
section 4 needs `trl`; each skips with a warning when its dependency is
absent, so this stays runnable in a bare checkout.

Section 4 builds a CPU-only `GRPOConfig`. Two constraints apply and both are
load-bearing, not stylistic: `use_cpu=True` is required because transformers
raises "Your setup doesn't support bf16/gpu" without it, and
`per_device_train_batch_size` must be divisible by `num_generations`, so the
batch size of 4 below pairs with the group size of 4.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "openenv-custom-training"))

_FAILURES = []


def _fail(msg):
    print(f"❌ {msg}")
    _FAILURES.append(msg)


def _section(name, fn, *args):
    """Run one section and report its own verdict, independent of earlier ones."""
    before = len(_FAILURES)
    fn(*args)
    if len(_FAILURES) == before:
        print(f"✅ {name} passed.")


def _load_simple_env():
    try:
        import env_simple_task  # noqa: F401
    except ImportError as exc:
        print(f"⚠️  {exc.name} not installed; skipping the environment contract tests.")
        return None
    return env_simple_task


def test_environment_contract(mod):
    """The environment_factory contract, checked structurally."""
    print("[Contract Test] Verifying SimpleGuessEnv honours environment_factory...")
    import inspect

    # __init__ takes no arguments beyond self — TRL calls EnvironmentFactory().
    params = list(inspect.signature(mod.SimpleGuessEnv).parameters)
    if params:
        _fail(f"SimpleGuessEnv.__init__ must take no arguments, takes {params}")

    env = mod.SimpleGuessEnv()

    # Every public method other than reset is a tool and needs a full Args:
    # block — transformers.get_json_schema raises DocstringParsingException
    # otherwise, and a one-line docstring is the usual way that happens.
    tools = [
        n for n in dir(env)
        if not n.startswith("_") and n != "reset" and callable(getattr(env, n))
    ]
    if not tools:
        _fail("SimpleGuessEnv exposes no tools")
    for name in tools:
        doc = inspect.getdoc(getattr(env, name))
        if not doc:
            _fail(f"tool {name} has no docstring")
            continue
        if "Args:" not in doc:
            _fail(f"tool {name} docstring has no Args: block")
            continue
        for param in inspect.signature(getattr(env, name)).parameters:
            if f"{param}:" not in doc:
                _fail(f"tool {name} docstring omits Args entry for {param!r}")

    # Episode state lives on the instance; reward funcs read it back off there.
    for attr in ("reward", "done"):
        if not hasattr(env, attr):
            _fail(f"SimpleGuessEnv has no `{attr}` attribute")


def test_episode_rollout(mod):
    """Drive episodes by hand: a solved one scores 1.0, a stalled one 0.0."""
    print("[Contract Test] Driving episodes by hand...")

    # Solved: bisect to the target.
    env = mod.SimpleGuessEnv()
    env.reset(target=42)
    low, high = 0, 99
    while not env.done:
        mid = (low + high) // 2
        feedback = env.guess(mid)
        if feedback == "higher":
            low = mid + 1
        elif feedback == "lower":
            high = mid - 1
    if env.reward != 1.0:
        _fail(f"a bisecting episode should score 1.0, scored {env.reward}")

    # Unsolvable: burn every attempt on the same wrong guess.
    env = mod.SimpleGuessEnv()
    env.reset(target=42)
    for _ in range(mod.MAX_ATTEMPTS):
        if env.done:
            break
        env.guess(7)
    if not env.done:
        _fail(f"episode did not terminate at MAX_ATTEMPTS={mod.MAX_ATTEMPTS}")
    if env.reward != 0.0:
        _fail(f"an unsolved episode should score 0.0, scored {env.reward}")

    # An invalid call is rejected by raising; TRL feeds the message back.
    env = mod.SimpleGuessEnv()
    env.reset(target=42)
    try:
        env.guess(500)
        _fail("an out-of-range guess should raise")
    except ValueError:
        pass


def test_reward_function(mod):
    """reward_func(environments, **kwargs) -> one float per environment."""
    print("[Contract Test] Verifying reward function logic...")
    solved, unsolved = mod.SimpleGuessEnv(), mod.SimpleGuessEnv()
    solved.reward, unsolved.reward = 1.0, 0.0

    rewards = mod.reward_func([solved, unsolved])
    if rewards != [1.0, 0.0]:
        _fail(f"Expected [1.0, 0.0], got {rewards}")
    elif len(set(rewards)) < 2:
        _fail("reward must vary with episode outcome or GRPO gets no advantage")


def test_grpo_config_contract():
    print("[Contract Test] Verifying GRPOConfig field compatibility...")
    try:
        from trl import GRPOConfig

        if not isinstance(GRPOConfig, type):
            print("⚠️  TRL is mocked; skipping class instantiation test.")
            return

        config = GRPOConfig(
            output_dir="./tmp_checkpoints",
            vllm_server_host="localhost",
            vllm_server_port=8000,
            learning_rate=5e-6,
            per_device_train_batch_size=4,
            num_generations=4,
            use_cpu=True,
        )
        assert config.vllm_server_host == "localhost"
        assert config.vllm_server_port == 8000

        # The regression this check exists for: `vllm_server_url` is not a
        # GRPOConfig field and crashes at construction.
        try:
            GRPOConfig(
                output_dir="./tmp_checkpoints",
                vllm_server_url="http://localhost:8000",
                learning_rate=5e-6,
                per_device_train_batch_size=4,
                num_generations=4,
                use_cpu=True,
            )
            _fail("GRPOConfig accepted vllm_server_url; that field must not exist")
        except TypeError:
            pass

        print("✅ GRPOConfig vLLM parameters compatibility passed.")
    except ImportError:
        print("⚠️  TRL package not installed in current env; skipping class instantiation test.")
    except Exception as e:
        _fail(f"GRPOConfig test failed: {e}")


def main():
    print("🚀 Starting OpenEnv RL Contract Verification...")
    mod = _load_simple_env()
    if mod is not None:
        _section("environment_factory structural contract", test_environment_contract, mod)
        _section("episode rollout contract", test_episode_rollout, mod)
        _section("reward calculation contract", test_reward_function, mod)
    test_grpo_config_contract()

    if _FAILURES:
        print(f"\n💥 {len(_FAILURES)} contract failure(s).")
        sys.exit(1)
    print("✨ All OpenEnv RL Contract Tests Passed Successfully!")


if __name__ == "__main__":
    main()
