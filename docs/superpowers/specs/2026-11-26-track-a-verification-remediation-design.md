# Track A Design: Verification Truthfulness Remediation

**Date:** 2026-11-26
**Branch:** `claude/track-a-verification-truthfulness`
**Scope:** `beer-sakthai/openenv-rl-training` (local clone; not pushed)

## 1. Motivation

This repository has an unusually disciplined evidence culture: a "written, not run"
labeling convention, verification claims that carry a date and a method, a parseable
gap ledger in `docs/KNOWN_GAPS.yaml`, and a PLAN -> CHANGELOG -> KNOWN_GAPS lifecycle
that keeps resolved work from lingering as an open item.

That culture is applied rigorously to the training and evaluation code and barely at
all to the rest of the repository. Three concrete failures follow from that gap, and
all three were confirmed by execution rather than by reading:

1. A live defect in the multi-catalog runner terminates every echo episode after one
   tool call, and the test guarding it cannot detect the defect.
2. The GRPOConfig contract check -- the only automated guard against the
   `vllm_server_url` regression class -- never executes in CI, and would fail
   immediately if it did.
3. The test counts, the published-model claims, and the artifact list in FINDINGS.md
   do not match measured or live state.

The 169-test figure deserves particular note. It appears in the README badge, the
README prose, three places in CLAUDE.md, and the 2026-11-21 CHANGELOG entry. The
measured count was 160 at the time of writing (171 after PR #249 -- see section 3.5),
and the commit that announced "expand test suite to 169 tests" already contained 160
tests. The number was never true at any commit.

## 2. Scope

### In scope

- The `or True` episode-termination defect and the test that fails to detect it.
- Making the GRPOConfig contract check genuinely execute, including the parameter
  repair that prerequisite demands.
- The documentation corrections that separate *published* checkpoints from *training
  targets*, and the stale test counts.
- A new CHANGELOG entry recording all of the above.
- This design document.

### Explicitly out of scope

- The GRPO training run itself, which requires GPU time or Hugging Face credit.
- The 16 agent-harness references that resolve to no file; that is Track C.
- The missing `openenv/repl` catalog environment, which requires an upstream change.
- Any change to the reward design, the Tier A / Tier B split, or the sandbox guards.
- Rewriting historical CHANGELOG entries. The 2026-11-21 claim of 169 tests remains as
  history; a new entry records the correction.
- Adding trl to the pytest environment (see section 5.3).

## 3. Verified evidence

Every claim below was established by execution in this session, not by reading.

### 3.1 The echo defect

`openenv-multi-catalog-training/train_multi_env.py:226` reads:

```python
self.done = result.observation.done or True
```

Python's `or` yields the first truthy operand, so the expression is unconditionally
`True` and the server's episode state is discarded. A mutation test isolates it:
setting the mock's `observation.done` to `False` and asserting `env.done is False`
fails against the current source with `assert True is False`, and passes once the
line reads `self.done = result.observation.done`.

The shipped test cannot catch it. `test_multi_catalog_env_step_tools_execution` sets
`echo_step.observation.done = True` before the call and then asserts `env.done is
True`, which holds under the buggy and the correct implementation alike. The
adjacent Sudoku case in the same function already uses the discriminating pattern
(`observation.done = False` -> `assert env.done is False`).

This is the same class of failure CLAUDE.md already names: "an earlier revision
asserted a bare-string `prompt` and an `env_outputs=` kwarg TRL never passes, and so
stayed green while both were bugs."

The three other unconditional `self.done = True` sites (`coding_run`, `repl_finish`)
encode deliberate one-shot semantics and are not in scope.

### 3.2 The GRPOConfig check

`verify_grpo_contract.py` check #4 constructs a `GRPOConfig` and skips when `trl` is
absent. CI installs only `datasets`, so the check prints
`TRL package not installed in current env; skipping class instantiation test` and
asserts nothing.

Installing trl as-is turns CI red. Against trl 1.18.1 with transformers 5.24.1 on
CPU:

```
❌ GRPOConfig test failed: generation_batch_size (1) must be divisible by num_generations (8).
```

Three construction requirements were established empirically:

| Requirement | Evidence |
|---|---|
| `use_cpu=True` is mandatory on a CPU box | otherwise `ValueError: Your setup doesn't support bf16/gpu. You need to assign use_cpu if you want to train the model on CPU.` |
| `per_device_train_batch_size` must be divisible by `num_generations` | `per_device_train_batch_size=1` with the default `num_generations=8` raises |
| `num_generations` must be at least 2 | `num_generations=1` raises `GRPO requires at least 2 generations per prompt to calculate the advantages.` |

The guard's premise also holds. `GRPOConfig(vllm_server_url=...)` raises
`TypeError: GRPOConfig.__init__() got an unexpected keyword argument
'vllm_server_url'`.

`use_cpu=True` is already present in the script and is load-bearing, not vestigial.
No `tmp_checkpoints` directory is created by the construction.

### 3.3 Install cost

`uv run --with` ignores `UV_TORCH_BACKEND` and silently resolves the CUDA torch
wheel, because `uv run` has no `--torch-backend` flag (it exists only on
`uv pip install`). Measured in this sandbox:

| Path | Result |
|---|---|
| `uv run --with trl==1.18.1` (implicit CUDA) | torch 2.15.0+cu132, plus nvidia-cublas, cudnn, nccl, cufft, cusolver, triton; **5m20s** |
| `uv pip install --torch-backend cpu torch` | torch 2.15.0+cpu; **2m26s** |

### 3.4 The conftest interaction

`conftest.py` prefers the real package and falls back to a permissive stub only on
ImportError:

```python
def _install_trl_stub():
    if "trl" not in sys.modules:
        try:
            import trl  # noqa: F401
        except ImportError:
            class GRPOConfig:
                def __init__(self, *args, **kwargs):
                    ...
```

The stub accepts any keyword argument; the real dataclass rejects unknown ones.
Adding trl to the pytest environment would therefore change how every test that
constructs a `GRPOConfig` behaves. That is a test-suite behaviour change this track
does not intend to make, so trl is kept in a separate environment.

### 3.5 Measured test counts

Measured 2026-11-26 by running the CI invocation on a pristine tree:

| Suite | Measured |
|---|---|
| root contract (`test_browsergym_contract.py`) | 15 |
| `sakthai-agentic-eval-train/tests/` | 30 |
| `openenv-custom-training/tests/` | 53 |
| `openenv-multi-catalog-training/tests/` | 25 |
| `sakthai-sft-training/tests/` | 37 |
| **total** | **160** |

> **Amended 2026-11-27.** PR #249 merged to `main` after this spec was written and
> added 11 tests (`openenv-custom-training` 53 -> 59, `openenv-multi-catalog-training`
> 25 -> 30), so the suite now collects **171**. The table above records the 2026-11-26
> measurement of `a65fb23`, the tree this design was written against; the live counts in
> README.md and CLAUDE.md carry the post-merge values (171 total; root 15, agentic-eval
> 30, custom 59, multi-catalog 30, SFT 37), re-measured on the rebased branch.

## 4. Design

### 4.1 The episode-termination fix

`openenv-multi-catalog-training/train_multi_env.py`, line 226:

```python
-        self.done = result.observation.done or True
+        self.done = result.observation.done
```

### 4.2 The discriminating test

`openenv-multi-catalog-training/tests/test_multi_catalog_env.py`. The existing echo
case is flipped so the assertion can distinguish the two implementations, and a
second echo case covers the terminal path:

```python
    echo_step.observation.echoed_message = "the falcon flies at dawn"
    echo_step.observation.done = False
    ...
    out_echo = env.echo_say("the falcon flies at dawn")
    assert out_echo == "the falcon flies at dawn"
    assert env.reward == 1.0
    assert env.done is False
```

A second echo instance with `observation.done = True` asserts `env.done is True` and
`env.reward == 1.0`.

The existing test function is **extended rather than duplicated**, so the suite
count stays at 160 and the counts written into the documentation remain true.

### 4.3 The contract check

`verify_grpo_contract.py`, check #4:

- `per_device_train_batch_size=1` becomes `per_device_train_batch_size=4`, with
  `num_generations=4` added. The group size of 4 matches the repository's default.
- `use_cpu=True` is retained.
- An explicit negative assertion is added: `GRPOConfig(vllm_server_url=...)` must
  raise `TypeError`.
- The `ImportError` skip is retained so the script stays runnable in a bare checkout.

### 4.4 CI wiring

`.github/workflows/verify-contracts.yml` replaces the contract step with:

```yaml
      - name: verify_grpo_contract.py (CPU; trl-backed GRPOConfig check)
        run: |
          uv venv .venv-verify
          uv pip install --python .venv-verify --torch-backend cpu torch
          uv pip install --python .venv-verify "trl==1.18.1" "transformers>=5.23.0" datasets
          .venv-verify/bin/python verify_grpo_contract.py
```

`trl` is pinned to `1.18.1` to match `openenv-custom-training/requirements.txt`. A
guard that validates a different TRL than the training code will use is worthless.
`transformers>=5.23.0` matches the same requirements file. (`5.2.0` is the floor
TRL itself enforces -- `GRPOTrainer` raises below it -- not the pin this repository
uses; the requirements file pins `>=5.23.0`.)

The pytest step is unchanged and deliberately does not gain trl (section 3.4).

### 4.5 Documentation corrections

**Test counts.** Every claim is replaced with the measured value and dated, so a
future drift is detectable rather than silent.

| Site | Current | Corrected |
|---|---|---|
| `README.md:9` badge | 169 | 160 |
| `README.md:97` heading | 149 Tests | 160 Tests |
| `README.md:71-74`, `:76` table | 36 / 47 / 24 / 29 / 12 | 37 / 53 / 25 / 30 / 15 |
| `README.md:118` inline comment | 34 tests | 25 tests |
| `README.md:243` | 169 pytest tests | 160 pytest tests |
| `CLAUDE.md:31` | 169 pytest tests | 160 pytest tests |
| `CLAUDE.md:76` | ruff + 94 CPU tests | ruff + 160 CPU tests |
| `CLAUDE.md:97` | # 130 passed | # 160 passed |

The README inline comments at lines 115, 121, 124 and 127 (37, 53, 30, 15) are
already correct and are left alone.

**Published checkpoints versus training targets.**
`Nanthasit/sakthai-context-7b-tools-grpo` and `Nanthasit/sakthai-context-0.5b-tools-sft`
both return HTTP 401 from the Hub API, and the author's published model list contains
no `-grpo` model at all. Four sites assert publication and must be corrected:

| Site | Correction |
|---|---|
| `README.md:28` | "Produces final RL-aligned checkpoints" becomes a statement that training runs target a repo which does not yet exist |
| `README.md:52` | "Production RL checkpoint aligned on OpenEnv tasks / Final output" becomes not-yet-produced |
| `docs/MODELS.md:15`, `:150` | "Production Aligned Checkpoint" and the deployed-flow diagram become a target |
| `FINDINGS.md:170`, `:175`, `:177`, `:179` | the "all on the Hub" heading is false for three of five items; `sft` becomes `sft-v2`; the two GRPO repos are recorded as not published |

**Left untouched deliberately.** Every `TRAIN_PUSH_TO=...-grpo` occurrence
(`CLAUDE.md:313`, `PLAN.md:35`, `README.md:219`, `FINDINGS.md:130`,
`sakthai-agentic-eval-train/README.md:70`) is a legitimate push target, as are
`docs/MODELS.md:45` and `docs/CONSOLIDATION.md:128`. Rewriting them would destroy
the roadmap's intent.

**CI and local-command descriptions.** `CLAUDE.md` currently describes CI as a
CPU-only toolchain without torch or trl, and documents the contract script's local
command. Both change to match the new recipe, including the statement at
`CLAUDE.md:20-22` that the root requirements file "is only the CPU dev/test
toolchain `verify-contracts.yml` installs".

**CHANGELOG.** A new dated entry at the top of `CHANGELOG.md` records the defect
fix, the strengthened test, the contract-check repair, the CI wiring, the count
corrections, and the fact that the 169 figure was never accurate. The historical
2026-11-21 entry is not rewritten.

## 5. Verification

Each item has an objective pass condition.

| Check | Command | Pass condition |
|---|---|---|
| Lint | `uv run --with ruff ruff check .` | All checks passed |
| Test suite | the CI five-suite invocation | 160 passed, no skips |
| Contract check, trl present | `.venv-verify/bin/python verify_grpo_contract.py` | all four sections pass |
| Contract check, trl absent | `uv run --with datasets python3 verify_grpo_contract.py` | sections 1-3 pass; section 4 skips cleanly |
| Negative guard | direct construction with `vllm_server_url` | raises `TypeError` |
| Defect fix | echo case with server `done=False` | `env.done is False` |
| Counts | grep the corrected sites | every claim equals the measured value |
| Publication claims | grep the four sites | none presents an unpublished model as published |
| Whitespace | `git diff --check` | clean |

Both contract-check paths must be exercised. The second is what keeps the script
usable in a bare checkout, and it is the path a contributor without a GPU box will
hit.

## 6. Commit strategy

Branch `claude/track-a-verification-truthfulness`, following the repository's
`claude/<topic>-<suffix>` convention. Commits use the repository's Conventional
Commits style and are scoped to the repository's git configuration only.

| Commit | Contents |
|---|---|
| `fix(train): respect server done state in MultiCatalogEnv.echo_say` | section 4.1 |
| `test(multi-catalog): make the echo done assertion discriminating` | section 4.2 |
| `fix(contract): make the GRPOConfig check runnable on CPU and assert vllm_server_url is rejected` | section 4.3 |
| `ci: install CPU torch and trl so the GRPOConfig contract check executes` | section 4.4 |
| `docs: correct test counts and distinguish published checkpoints from training targets` | section 4.5 |
| `docs: add the Track A design spec` | this document |

The branch is not pushed. Pushing requires credentials that are not available in the
working environment.

## 7. Risks

| Risk | Assessment |
|---|---|
| CI duration increases by roughly two and a half minutes per run | Accepted. The guard protects a regression class that has already produced three manual fixes. Caching the venv is a later optimisation. |
| The trl pin drifts from `openenv-custom-training/requirements.txt` | Mitigated by pinning both to `1.18.1` in this change; a future bump must move both together. |
| The real `trl` changes behaviour if it is ever added to the pytest environment | Avoided by construction; section 3.4 explains why the pytest step stays unchanged. |
| The 160 count drifts again as tests are added | Mitigated by dating every count claim and by adding a CHANGELOG entry, which is the repository's existing mechanism. |
| The defect fix changes training behaviour | It restores intended behaviour. `echo_say` previously could never continue past one call regardless of server state. |

## 8. Out of scope for this track, tracked elsewhere

- The GRPO run that would produce a published RL checkpoint: blocked on Hugging Face
  Jobs credit, and on the browsergym wrapper remaining unverified.
- The `openenv/repl` catalog environment that `run_servers.sh` pulls but the upstream
  organization does not publish.
- The 16 agent-harness references that resolve to no file, and the three
  SKILL.md-only skills.
- The two unexecuted skill evaluations.
