#!/usr/bin/env python3
"""CPU-only static + light-dynamic validator for a generated env_factory module.

Catches the failure modes that would otherwise blow up at trainer
construction (or worse, silently mid-run). Runs offline — no torch, no trl,
no vllm — so it's cheap to call after every scaffold.

Usage:
    python scripts/validate_env.py <path/to/env_module.py>

Exit codes:
    0 — all checks pass
    1 — one or more checks failed (details on stderr/stdout)
    2 — the file itself did not import (syntax error, missing dep)
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import re
import sys
import textwrap
import types
from pathlib import Path

# Fields TRL / the openenv server / train.py check for. Failing any of these
# tanks a real run silently or at construction time.
REQUIRED_MODULE_EXPORTS = ("ENVIRONMENT_FACTORY", "REWARD_FUNCS", "TRAIN_DATASET")
DEFAULT_MODEL_MERGED_TRAP = re.compile(r"sakthai-context-[\d.]+b-merged")


class Result:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def note(self, msg: str) -> None:
        self.info.append(msg)

    def report(self) -> int:
        for msg in self.info:
            print(f"  ok   {msg}")
        for msg in self.warnings:
            print(f"  warn {msg}", file=sys.stderr)
        for msg in self.errors:
            print(f"  ERR  {msg}", file=sys.stderr)
        if self.errors:
            print(f"\nFAILED: {len(self.errors)} error(s).", file=sys.stderr)
            return 1
        print(f"\nOK: {len(self.info)} check(s) passed, {len(self.warnings)} warning(s).")
        return 0


def import_module(path: Path) -> types.ModuleType:
    """Import the target module in-process. Mocks the heavy training stack
    so the file is importable without torch/trl/vllm/openenv installed —
    the actual scaffolds only reach those symbols at runtime, not import.
    """
    from unittest import mock

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


# --- Individual checks ------------------------------------------------------

def check_module_exports(module: types.ModuleType, result: Result) -> None:
    for name in REQUIRED_MODULE_EXPORTS:
        if hasattr(module, name):
            result.note(f"exports {name}")
        else:
            result.error(
                f"missing module-level export `{name}` — train.py's dispatch "
                f"table looks these names up by convention"
            )
    if hasattr(module, "DEFAULT_MODEL"):
        model = getattr(module, "DEFAULT_MODEL")
        if isinstance(model, str) and DEFAULT_MODEL_MERGED_TRAP.search(model):
            result.error(
                f"DEFAULT_MODEL={model!r} is a *-merged base — those ship no "
                f"chat template with tool-calling and GRPOTrainer will raise. "
                f"Use sakthai-context-{{1.5b,7b}}-tools instead."
            )
        else:
            result.note(f"DEFAULT_MODEL = {model!r}")
    else:
        result.warning("no DEFAULT_MODEL — train.py's --model flag will not have a default")


def check_env_class(env_cls, result: Result) -> None:
    if not inspect.isclass(env_cls):
        result.error(f"ENVIRONMENT_FACTORY is not a class: {env_cls!r}")
        return
    result.note(f"ENVIRONMENT_FACTORY class: {env_cls.__name__}")

    init_sig = inspect.signature(env_cls.__init__)
    params = [p for p in init_sig.parameters.values() if p.name != "self"]
    if params:
        result.error(
            f"{env_cls.__name__}.__init__ takes arguments {[p.name for p in params]}; "
            f"the contract requires __init__(self) — TRL instantiates as "
            f"EnvironmentFactory(), never EnvironmentFactory(cfg)"
        )
    else:
        result.note(f"{env_cls.__name__}.__init__ takes no args")

    if not hasattr(env_cls, "reset"):
        result.error(f"{env_cls.__name__} has no reset() method")

    # Public tool methods: everything callable and public other than reset.
    tool_methods = [
        name for name, obj in inspect.getmembers(env_cls, predicate=inspect.isfunction)
        if not name.startswith("_") and name != "reset"
    ]
    if not tool_methods:
        result.warning(
            f"{env_cls.__name__} exposes no public tool methods — reset only. "
            f"Is that intentional?"
        )
    else:
        result.note(f"tool methods: {tool_methods}")

    # Instantiation + reward/done initialisation.
    try:
        env = env_cls()
    except Exception as exc:
        result.error(f"{env_cls.__name__}() raised at instantiation: {exc}")
        return
    for attr in ("reward", "done"):
        if not hasattr(env, attr):
            result.error(
                f"{env_cls.__name__}.{attr} is not set in __init__ — the "
                f"reward function reads it back; a never-stepped rollout will "
                f"AttributeError instead of returning 0.0"
            )

    check_tool_docstrings(env_cls, tool_methods, result)


def check_tool_docstrings(env_cls, tool_methods: list[str], result: Result) -> None:
    """Every public tool method needs a Google-style docstring with an
    Args: block covering every parameter. Missing Args: raises
    DocstringParsingException at trainer construction.
    """
    for name in tool_methods:
        fn = getattr(env_cls, name)
        doc = inspect.getdoc(fn) or ""
        if not doc:
            result.error(f"tool `{name}` has no docstring — will fail get_json_schema")
            continue
        if "\n" not in doc.strip():
            result.error(
                f"tool `{name}` has a one-line docstring — needs Google-style with "
                f"an Args: block"
            )
            continue
        sig = inspect.signature(fn)
        params = [p.name for p in sig.parameters.values() if p.name != "self"]
        if params:
            if "Args:" not in doc and "Arguments:" not in doc:
                result.error(
                    f"tool `{name}` docstring has no Args: block — get_json_schema "
                    f"will raise DocstringParsingException at trainer construction"
                )
                continue
            for p in params:
                if not re.search(rf"^\s*{re.escape(p)}\s*[:\(]", doc, flags=re.M):
                    result.error(
                        f"tool `{name}` parameter `{p}` missing from Args: block"
                    )
        if not sig.parameters or all(
            p.annotation is inspect.Parameter.empty
            for p in sig.parameters.values() if p.name != "self"
        ):
            if params:
                result.warning(
                    f"tool `{name}` has no type hints — get_json_schema needs them "
                    f"to infer argument types"
                )
        else:
            result.note(f"tool `{name}` has docstring + Args + type hints")


def check_reward_func(reward_fn, result: Result) -> None:
    if not callable(reward_fn):
        # Might be a list of reward funcs in multi-env; check each.
        if isinstance(reward_fn, (list, tuple)):
            for fn in reward_fn:
                check_reward_func(fn, result)
            return
        result.error(f"REWARD_FUNCS is not callable: {reward_fn!r}")
        return
    sig = inspect.signature(reward_fn)
    params = list(sig.parameters.values())
    if not params or params[0].name != "environments":
        result.error(
            f"reward function `{reward_fn.__name__}` signature must be "
            f"(environments, **kwargs) -> list[float]; got "
            f"{list(sig.parameters.keys())}"
        )
    else:
        result.note(f"reward function `{reward_fn.__name__}` signature ok")


def check_dataset_builder(build_fn, result: Result) -> None:
    if not callable(build_fn):
        result.error(f"TRAIN_DATASET is not callable: {build_fn!r}")
        return
    try:
        ds = build_fn(4)
    except Exception as exc:
        result.error(f"TRAIN_DATASET(4) raised: {exc}")
        return
    try:
        row = ds[0]
    except Exception as exc:
        result.error(f"TRAIN_DATASET(4)[0] raised: {exc}")
        return
    if "prompt" not in row:
        result.error("dataset row missing `prompt` column")
        return
    prompt = row["prompt"]
    if not (isinstance(prompt, list) and prompt and isinstance(prompt[0], dict)
            and "role" in prompt[0] and "content" in prompt[0]):
        result.error(
            f"prompt is not a list of {{role, content}} dicts — TRL does "
            f"prompt[-1]['content'] and a bare string crashes with TypeError "
            f"on the first batch. Got: {type(prompt).__name__} = {prompt!r:.120}"
        )
    else:
        result.note(f"dataset has {len(ds)} rows; prompt is well-formed")


def check_tier_b_class_attr(path: Path, result: Result) -> None:
    """Tier B trap: SUPPORTS_CONCURRENT_SESSIONS at module level not on the
    class. Parse the file AST — importing may have set it on a MagicMock.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    module_level_names = {
        n.id for stmt in tree.body if isinstance(stmt, ast.Assign)
        for tgt in stmt.targets if isinstance(tgt, ast.Name) for n in [tgt]
    }
    if "SUPPORTS_CONCURRENT_SESSIONS" in module_level_names:
        result.error(
            "SUPPORTS_CONCURRENT_SESSIONS is declared at MODULE level; the "
            "server does getattr(env_cls, ...) on it and will refuse "
            "max_concurrent_envs > 1. Move it inside the Environment subclass."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="path to the env module to validate")
    args = parser.parse_args(argv)

    if not args.path.exists():
        print(f"no such file: {args.path}", file=sys.stderr)
        return 2

    print(f"Validating {args.path} …")
    result = Result()

    try:
        module = import_module(args.path)
    except Exception as exc:
        print(f"import failed: {exc}", file=sys.stderr)
        return 2

    check_module_exports(module, result)
    if hasattr(module, "ENVIRONMENT_FACTORY"):
        check_env_class(module.ENVIRONMENT_FACTORY, result)
    if hasattr(module, "REWARD_FUNCS"):
        check_reward_func(module.REWARD_FUNCS, result)
    if hasattr(module, "TRAIN_DATASET"):
        check_dataset_builder(module.TRAIN_DATASET, result)

    check_tier_b_class_attr(args.path, result)

    return result.report()


if __name__ == "__main__":
    raise SystemExit(main())
