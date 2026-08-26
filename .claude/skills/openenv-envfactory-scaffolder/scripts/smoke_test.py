#!/usr/bin/env python3
"""Drive one episode of a generated env module — catch runtime crashes the
static validator can't see.

Usage:
    python scripts/smoke_test.py <path/to/env_module.py>

What it does:
    1. Import the module (mocking trl / openenv / vllm so it loads on CPU).
    2. Build a small dataset (n=4).
    3. Instantiate the env.
    4. Call reset(**first_row_kwargs), print the initial observation.
    5. Call each public tool method once with a plausible argument value
       (dispatched by type hint), print the return.
    6. Check env.reward and env.done are read-back-able.

This does NOT run GRPO. It just proves the env's public surface behaves
without crashing before you hand it to the trainer.

Exit codes:
    0 — smoke test ran without unhandled exceptions
    1 — something crashed; the trace is printed
    2 — file did not import or has no ENVIRONMENT_FACTORY / TRAIN_DATASET
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
import traceback
import types
from pathlib import Path
from unittest import mock


PLAUSIBLE_BY_TYPE = {
    int: 0,
    float: 0.0,
    str: "smoke",
    bool: False,
    list: [],
    dict: {},
}


def import_module(path: Path) -> types.ModuleType:
    for name in ("trl", "openenv", "openenv.core", "openenv.core.env_server",
                 "openenv.core.env_server.interfaces",
                 "openenv.core.env_server.types",
                 "vllm"):
        sys.modules.setdefault(name, mock.MagicMock())
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def plausible_value(param: inspect.Parameter):
    """Pick a plausible argument value from the type hint. Falls back to 0."""
    ann = param.annotation
    if ann is inspect.Parameter.empty:
        return 0
    origin = getattr(ann, "__origin__", None)
    if origin in (list, dict):
        return PLAUSIBLE_BY_TYPE[origin]
    return PLAUSIBLE_BY_TYPE.get(ann, 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="path to the env module")
    args = parser.parse_args(argv)

    if not args.path.exists():
        print(f"no such file: {args.path}", file=sys.stderr)
        return 2

    print(f"Smoke-testing {args.path} …\n")

    try:
        module = import_module(args.path)
    except Exception as exc:
        print(f"import failed: {exc}", file=sys.stderr)
        return 2

    for name in ("ENVIRONMENT_FACTORY", "TRAIN_DATASET"):
        if not hasattr(module, name):
            print(f"module missing {name}", file=sys.stderr)
            return 2

    env_cls = module.ENVIRONMENT_FACTORY
    build_dataset = module.TRAIN_DATASET

    try:
        ds = build_dataset(4)
    except Exception:
        traceback.print_exc()
        print("dataset build failed", file=sys.stderr)
        return 1

    try:
        row = ds[0]
    except Exception:
        traceback.print_exc()
        return 1

    # Pass every column EXCEPT `prompt` (which the model would see) and
    # `environment` (dict-form multi-env's routing control) as kwargs.
    reset_kwargs = {k: v for k, v in row.items() if k not in ("prompt", "environment")}

    try:
        env = env_cls()
    except Exception:
        traceback.print_exc()
        print("env instantiation failed", file=sys.stderr)
        return 1
    print(f"instantiated {env_cls.__name__}()")
    print(f"initial state: reward={getattr(env, 'reward', '<missing>')} "
          f"done={getattr(env, 'done', '<missing>')}")

    try:
        obs = env.reset(**reset_kwargs)
    except Exception:
        traceback.print_exc()
        print(f"reset(**{reset_kwargs!r}) failed", file=sys.stderr)
        return 1
    print(f"\nreset({reset_kwargs!r}):")
    print(f"  → {obs!r:.200}")

    tool_methods = [
        (name, obj) for name, obj in inspect.getmembers(env_cls, predicate=inspect.isfunction)
        if not name.startswith("_") and name != "reset"
    ]
    if not tool_methods:
        print("\n(no public tool methods to smoke)")
    else:
        print(f"\nsmoking {len(tool_methods)} tool method(s):")

    for tool_name, tool_fn in tool_methods:
        sig = inspect.signature(tool_fn)
        arg_kwargs = {
            p.name: plausible_value(p)
            for p in sig.parameters.values() if p.name != "self"
        }
        try:
            result = tool_fn(env, **arg_kwargs)
            print(f"  {tool_name}({arg_kwargs!r}) → {result!r:.160}")
        except Exception as exc:
            print(f"  {tool_name}({arg_kwargs!r}) RAISED: {type(exc).__name__}: {exc}")
            # Raising is a valid outcome for out-of-range args, so this is
            # not a smoke-test failure by itself. Just log and continue.

    print(f"\nfinal state: reward={getattr(env, 'reward', '<missing>')} "
          f"done={getattr(env, 'done', '<missing>')}")
    print("\nsmoke test OK — no unhandled crashes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
