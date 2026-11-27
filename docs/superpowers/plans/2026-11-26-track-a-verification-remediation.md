# Track A Verification Truthfulness Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix a live episode-termination defect, make the GRPOConfig contract check actually execute in CI, and correct every stale test-count and published-model claim so the repository's numbers match measured and live state.

**Architecture:** Three code-bearing changes (one source fix with its test, one contract-script repair, one CI workflow change) followed by three documentation changes and a closing CHANGELOG entry. The contract check gains a valid CPU parameterization so that installing `trl` in CI produces a passing assertion rather than a red build, and it gains an explicit negative assertion so the `vllm_server_url` regression class cannot return silently. `trl` is installed in a dedicated virtual environment used only by the contract script, because `conftest.py` prefers the real `trl` package over its permissive stub and adding it to the pytest environment would change test-suite behaviour.

**Tech Stack:** Python 3.11, TRL 1.18.1, transformers 5.24.1, PyTorch 2.15.0+cpu, uv 0.13.9, pytest, ruff, GitHub Actions.

## Global Constraints

- Python 3.11 (`requires-python >=3.11` for trl 1.18.1).
- `trl` is pinned to exactly `1.18.1`, matching `openenv-custom-training/requirements.txt`. A guard that validates a different TRL than the training code will use is worthless.
- `transformers>=5.23.0`, matching the same requirements file. (`5.2.0` is the floor TRL itself enforces -- `GRPOTrainer` raises below it -- not the pin this repository uses.)
- PyTorch is installed CPU-only via `uv pip install --torch-backend cpu torch`. `uv run --with` ignores `UV_TORCH_BACKEND` and silently resolves the CUDA wheel.
- The pytest environment must NOT gain `trl`. `conftest.py` imports the real package when it is importable and falls back to a permissive stub only on ImportError; the real dataclass rejects unknown keyword arguments.
- The pytest suite must not gain tests from this plan's own changes. The echo test is extended, never duplicated, so the counts written into README.md and CLAUDE.md stay true.
- Branch: `claude/track-a-verification-truthfulness`. Never commit to `main`.
- Commit identity is scoped to this repository: `beer-sakthai <293853443+beer-sakthai@users.noreply.github.com>`.
- Commit subjects follow Conventional Commits with the repository's optional scope style, and carry the trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- The branch is never pushed. No push credentials exist in this environment.
- Historical CHANGELOG entries are never rewritten. Corrections are added as new entries.
- `TRAIN_PUSH_TO=...-grpo` occurrences are push targets and must not be edited.

> **Amended 2026-11-27.** Two things changed after this plan was written, and both are
> recorded here rather than silently absorbed:
>
> 1. **PR #249 merged to `main`** and added 11 tests (`openenv-custom-training` 53 -> 59,
>    `openenv-multi-catalog-training` 25 -> 30), so the suite now collects **171**, not
>    160. Every "160" quoted in this plan records the 2026-11-26 measurement of
>    `a65fb23`, the tree the plan was written against. The branch was rebased onto
>    `main` after that merge, and the live counts in README.md and CLAUDE.md carry the
>    post-merge values (171 total; root 15, agentic-eval 30, custom 59, multi-catalog
>    30, SFT 37), re-measured on the rebased tree.
> 2. **The `transformers` pin was corrected from `5.2.0` to `5.23.0`.** `5.2.0` is the
>    floor TRL itself enforces (see the explanatory comment in
>    `openenv-custom-training/requirements.txt`); the requirements file pins
>    `transformers>=5.23.0`, and the implementation matches it.
> 3. **The branch was pushed and a pull request opened on 2026-11-27**, superseding the
>    "never pushed" constraint above. That constraint recorded the state of the
>    environment at planning time — no push credentials existed, so the only route to
>    GitHub was the connected GitHub app, which became available for the save step.

---

## File Structure

| File | Change | Responsibility after this plan |
|---|---|---|
| `openenv-multi-catalog-training/train_multi_env.py` | Modify line 226 | Owns episode state per catalog environment. The `echo_say` tool now mirrors the server's `done` instead of forcing termination. |
| `openenv-multi-catalog-training/tests/test_multi_catalog_env.py` | Modify the echo case | Guards that `MultiCatalogEnv` mirrors the server's episode state in both directions. |
| `verify_grpo_contract.py` | Modify check #4 and the module docstring | The CPU-only contract gate. Section 4 now constructs a valid `GRPOConfig` on CPU and asserts that `vllm_server_url` is rejected. |
| `.github/workflows/verify-contracts.yml` | Modify the contract step and the job timeout | CI gate. Installs CPU torch and pinned trl into a dedicated venv, then runs the contract script. |
| `.gitignore` | Add one entry | Keeps the CI-created `.venv-verify/` out of `git status` for anyone running the recipe locally. |
| `README.md` | Modify count claims, the published-checkpoint bullet, the capability matrix row, and the local command | The public front door. Counts carry a date; published models are distinguished from training targets. |
| `CLAUDE.md` | Modify count claims, the CI toolchain description, and the local command | The operating manual for contributors and agents. Same truthfulness standard as the README. |
| `docs/MODELS.md` | Modify the model table row and the orchestration diagram | The 18-model portfolio reference. Marks the GRPO checkpoint as a target. |
| `sakthai-agentic-eval-train/FINDINGS.md` | Modify the artifacts heading and three artifact bullets | The durable empirical record. Records what was produced versus what was published. |
| `CHANGELOG.md` | Add a dated entry | The landed-work log. Records the defect fix, the guard, and the count corrections. |
| `docs/superpowers/plans/2026-11-26-track-a-verification-remediation.md` | Created by this plan | The implementation plan itself. |

---

### Task 1: Echo episode termination follows the server

**Files:**
- Modify: `openenv-multi-catalog-training/train_multi_env.py:226`
- Modify: `openenv-multi-catalog-training/tests/test_multi_catalog_env.py:215-227`

**Interfaces:**
- Consumes: `MultiCatalogEnv.echo_say(message: str) -> str`, which returns the echoed message and mutates `self.reward` and `self.done`.
- Produces: `MultiCatalogEnv.echo_say` sets `self.done` to exactly `result.observation.done`. No later task depends on this signature; the change is behavioural only.

- [ ] **Step 1: Write the failing test**

In `openenv-multi-catalog-training/tests/test_multi_catalog_env.py`, replace the echo block:

```python
    # echo_say
    env = MultiCatalogEnv()
    env.active = "echo"
    env._secret_phrase = "the falcon flies at dawn"
    echo_client = MagicMock()
    echo_step = MagicMock()
    echo_step.observation.echoed_message = "the falcon flies at dawn"
    echo_step.observation.done = True
    echo_client.step.return_value = echo_step
    env._clients["echo"] = echo_client
    out_echo = env.echo_say("the falcon flies at dawn")
    assert out_echo == "the falcon flies at dawn"
    assert env.reward == 1.0
    assert env.done is True
```

with:

```python
    # echo_say — a non-terminal server response must not end the episode.
    env = MultiCatalogEnv()
    env.active = "echo"
    env._secret_phrase = "the falcon flies at dawn"
    echo_client = MagicMock()
    echo_step = MagicMock()
    echo_step.observation.echoed_message = "the falcon flies at dawn"
    echo_step.observation.done = False
    echo_client.step.return_value = echo_step
    env._clients["echo"] = echo_client
    out_echo = env.echo_say("the falcon flies at dawn")
    assert out_echo == "the falcon flies at dawn"
    assert env.reward == 1.0
    assert env.done is False

    # echo_say — a terminal server response ends the episode.
    env = MultiCatalogEnv()
    env.active = "echo"
    env._secret_phrase = "the falcon flies at dawn"
    echo_client = MagicMock()
    echo_step = MagicMock()
    echo_step.observation.echoed_message = "the falcon flies at dawn"
    echo_step.observation.done = True
    echo_client.step.return_value = echo_step
    env._clients["echo"] = echo_client
    out_echo = env.echo_say("the falcon flies at dawn")
    assert out_echo == "the falcon flies at dawn"
    assert env.reward == 1.0
    assert env.done is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
export UV_INDEX_URL=https://pypi.org/simple UV_DEFAULT_INDEX=https://pypi.org/simple
uv run --with datasets --with pytest --with pydantic --with httpx --with anyio \
  pytest openenv-multi-catalog-training/tests/test_multi_catalog_env.py::test_multi_catalog_env_step_tools_execution -q
```

Expected: FAIL with `assert True is False` on the `assert env.done is False` line, because the source still contains `or True`.

- [ ] **Step 3: Write the minimal implementation**

In `openenv-multi-catalog-training/train_multi_env.py`, change line 226:

```python
-        self.done = result.observation.done or True
+        self.done = result.observation.done
```

Do not touch `coding_run` or `repl_finish`. Their unconditional `self.done = True` encodes deliberate one-shot semantics.

- [ ] **Step 4: Run the test to verify it passes**

Run the command from Step 2.

Expected: PASS.

- [ ] **Step 5: Run the full suite and lint**

Run:

```bash
export UV_INDEX_URL=https://pypi.org/simple UV_DEFAULT_INDEX=https://pypi.org/simple
uv run --with ruff ruff check .
uv run --with datasets --with pytest --with pydantic --with httpx --with anyio \
  pytest test_browsergym_contract.py \
         sakthai-agentic-eval-train/tests/ \
         openenv-custom-training/tests/ \
         openenv-multi-catalog-training/tests/ \
         sakthai-sft-training/tests/ -q
```

Expected: `All checks passed!` and `160 passed`.

- [ ] **Step 6: Commit**

```bash
git add openenv-multi-catalog-training/train_multi_env.py
git commit -m "fix(train): respect server done state in MultiCatalogEnv.echo_say

\`self.done = result.observation.done or True\` is unconditionally True,
because Python's \`or\` yields the first truthy operand. The server's
episode state was discarded and every echo episode ended after a single
tool call.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git add openenv-multi-catalog-training/tests/test_multi_catalog_env.py
git commit -m "test(multi-catalog): make the echo done assertion discriminating

The echo case asserted \`env.done is True\` against a mock already
reporting \`done = True\`, so it passed under both the buggy and the
correct implementation. It now asserts the environment follows the
server in the non-terminal case, and keeps a terminal case alongside.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: The GRPOConfig contract check runs on CPU and guards the regression

**Files:**
- Modify: `verify_grpo_contract.py:15-20` (module docstring)
- Modify: `verify_grpo_contract.py:146-169` (`test_grpo_config_contract`)

**Interfaces:**
- Consumes: `trl.GRPOConfig` from the environment the script runs in.
- Produces: `test_grpo_config_contract()` constructs a valid CPU `GRPOConfig` and asserts that `vllm_server_url` raises `TypeError`. The function keeps its existing signature and its existing `ImportError` skip.

- [ ] **Step 1: Run the check to verify it currently fails**

Run:

```bash
/workspace/.venv-trl/bin/python verify_grpo_contract.py
```

Expected: sections 1 to 3 pass, then

```
❌ GRPOConfig test failed: generation_batch_size (1) must be divisible by num_generations (8).
```

- [ ] **Step 2: Update the module docstring**

In `verify_grpo_contract.py`, replace lines 15 to 20:

```python
4. `GRPOConfig` accepts `vllm_server_host`/`vllm_server_port` (the fields that
   actually exist; `vllm_server_url` crashes at construction).

Sections 1-3 need `datasets` (env_simple_task imports it at module level) and
section 4 needs `trl`; each skips with a warning when its dependency is
absent, so this stays runnable in a bare checkout.
```

with:

```python
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
```

- [ ] **Step 3: Repair the construction and add the negative assertion**

In `verify_grpo_contract.py`, replace the body of `test_grpo_config_contract` from the `config = GRPOConfig(` line through the `print("✅ GRPOConfig vLLM parameters compatibility passed.")` line:

```python
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
```

The inner `try` wraps only the `GRPOConfig(...)` call, so a `TypeError` from the outer construction still reaches the outer `except Exception` handler and is reported as a failure.

- [ ] **Step 4: Run the check to verify it passes**

Run:

```bash
/workspace/.venv-trl/bin/python verify_grpo_contract.py
```

Expected: all four sections pass and the script exits with `✨ All OpenEnv RL Contract Tests Passed Successfully!`.

- [ ] **Step 5: Verify the bare-checkout path still skips cleanly**

Run:

```bash
export UV_INDEX_URL=https://pypi.org/simple UV_DEFAULT_INDEX=https://pypi.org/simple
uv run --with datasets python3 verify_grpo_contract.py
```

Expected: sections 1 to 3 pass, then `⚠️  TRL package not installed in current env; skipping class instantiation test.`, then `✨ All OpenEnv RL Contract Tests Passed Successfully!`. The script must exit 0.

- [ ] **Step 6: Run lint**

Run `uv run --with ruff ruff check .`

Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add verify_grpo_contract.py
git commit -m "fix(contract): make the GRPOConfig check runnable on CPU

The fourth check never executed in CI, because CI installs only datasets
and the check skips when trl is absent. Enabling it surfaced two more
problems: GRPOConfig requires use_cpu=True on a CPU box, and
per_device_train_batch_size must be divisible by num_generations.

The check now constructs with per_device_train_batch_size=4 and
num_generations=4, and explicitly asserts that vllm_server_url is
rejected with TypeError.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: CI installs the contract check's dependencies

**Files:**
- Modify: `.github/workflows/verify-contracts.yml:22` (job timeout)
- Modify: `.github/workflows/verify-contracts.yml:42-43` (contract step)
- Modify: `.gitignore` (append one entry)

**Interfaces:**
- Consumes: the repaired `verify_grpo_contract.py` from Task 2.
- Produces: a CI job whose contract step installs CPU-only torch and pinned trl into `.venv-verify/` and runs `.venv-verify/bin/python verify_grpo_contract.py`.

- [ ] **Step 1: Update the CI contract step**

In `.github/workflows/verify-contracts.yml`, replace:

```yaml
      - name: verify_grpo_contract.py (CPU, no torch/trl required)
        run: uv run --with datasets python3 verify_grpo_contract.py
```

with:

```yaml
      - name: verify_grpo_contract.py (CPU; trl-backed GRPOConfig check)
        run: |
          uv venv .venv-verify
          uv pip install --python .venv-verify --torch-backend cpu torch
          uv pip install --python .venv-verify "trl==1.18.1" "transformers>=5.23.0" datasets
          .venv-verify/bin/python verify_grpo_contract.py
```

The pytest step below it must keep its own dependency list. Adding `trl` there would replace the permissive `conftest.py` stub with the real dataclass and change how every test that constructs a `GRPOConfig` behaves.

- [ ] **Step 2: Raise the job timeout**

In the same file, change the job-level timeout:

```yaml
    timeout-minutes: 10
```

to:

```yaml
    timeout-minutes: 15
```

The CPU torch install measured 2m26s locally and the CUDA wheel measured 5m20s; a cold runner is slower, and the job also runs ruff, the baseline competition, and the five pytest suites.

- [ ] **Step 3: Ignore the verification venv**

Append to `.gitignore`:

```

# CI contract-check virtualenv (created by verify-contracts.yml)
.venv-verify/
```

- [ ] **Step 4: Verify the CI recipe locally**

Run, from the repository root:

```bash
export UV_INDEX_URL=https://pypi.org/simple UV_DEFAULT_INDEX=https://pypi.org/simple
uv venv .venv-verify
uv pip install --python .venv-verify --torch-backend cpu torch
uv pip install --python .venv-verify "trl==1.18.1" "transformers>=5.23.0" datasets
.venv-verify/bin/python verify_grpo_contract.py
```

Expected: `✨ All OpenEnv RL Contract Tests Passed Successfully!`

- [ ] **Step 5: Confirm the venv is ignored**

Run `git status --porcelain`

Expected: no output. The `.venv-verify/` directory must not appear.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/verify-contracts.yml .gitignore
git commit -m "ci: install CPU torch and trl so the GRPOConfig contract check executes

The contract step installed only datasets, so the fourth check skipped
and asserted nothing. It now installs CPU-only torch plus the pinned
trl the training workspace uses, into a dedicated venv so the pytest
environment keeps its conftest stubs.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Test-count claims match measured state

**Files:**
- Modify: `README.md:9`, `README.md:71-74`, `README.md:76`, `README.md:97`, `README.md:118`, `README.md:243`
- Modify: `CLAUDE.md:31`, `CLAUDE.md:76`, `CLAUDE.md:97`

**Interfaces:**
- Consumes: nothing. This task only reconciles prose with the measured counts.
- Produces: every count claim in README.md and CLAUDE.md equals the measured value of 160 total, dated 2026-11-26.

- [ ] **Step 1: Update the README badge and prose**

In `README.md`, replace the badge line:

```markdown
[![Tests](https://img.shields.io/badge/tests-169%20passed-brightgreen.svg)](tests)
```

with:

```markdown
[![Tests](https://img.shields.io/badge/tests-160%20passed-brightgreen.svg)](tests)
```

- [ ] **Step 2: Update the README section heading**

In `README.md`, replace:

```markdown
### 1. Comprehensive Test Suite (149 Tests in Single Invocation)
```

with:

```markdown
### 1. Comprehensive Test Suite (160 Tests in Single Invocation)
```

- [ ] **Step 3: Update the README workspace table counts**

In `README.md`, change the `Tests` column of the five workspace rows. Each row's cell is the last column of the table row.

| Row | Current cell | New cell |
|---|---|---|
| `sakthai-sft-training/` | `36 tests (...)` | `37 tests (...)` |
| `openenv-custom-training/` | `47 tests (\`tests/test_*.py\`)` | `53 tests (\`tests/test_*.py\`)` |
| `openenv-multi-catalog-training/` | `24 tests (...)` | `25 tests (...)` |
| `sakthai-agentic-eval-train/` | `29 tests (\`tests/test_*.py\`)` | `30 tests (\`tests/test_*.py\`)` |
| `Repo Root & CI` | `12 contract tests` | `15 contract tests` |

The parenthetical file lists in the first, third, and fourth rows are already correct and stay unchanged. The `browsergym-space/` row reads `Validated via contract` and is not a count; leave it alone.

- [ ] **Step 4: Update the README inline runner comment**

In `README.md`, replace:

```bash
# Test multi-catalog environment and A2A protocol agent (34 tests)
```

with:

```bash
# Test multi-catalog environment and A2A protocol agent (25 tests)
```

Leave the other four inline comments alone. They already read 37, 53, 30, and 15 and are correct.

- [ ] **Step 5: Update the README CI sentence**

In `README.md`, replace:

```markdown
- **CI Workflow**: [`.github/workflows/verify-contracts.yml`](.github/workflows/verify-contracts.yml) runs ruff, contract verifications, baseline competitions, and all 169 pytest tests across 5 workspace suites on every push and PR.
```

with:

```markdown
- **CI Workflow**: [`.github/workflows/verify-contracts.yml`](.github/workflows/verify-contracts.yml) runs ruff, contract verifications, baseline competitions, and all 160 pytest tests across 5 workspace suites on every push and PR.
```

- [ ] **Step 6: Update the three CLAUDE.md counts**

In `CLAUDE.md`, replace:

```markdown
competition & reward variance check), and 169 pytest tests across the root contract file and all
```

with:

```markdown
competition & reward variance check), and 160 pytest tests across the root contract file and all
```

In `CLAUDE.md`, replace:

```markdown
| `.github/workflows/` | `verify-contracts.yml` (ruff + 94 CPU tests, runs in GH Actions);
```

with:

```markdown
| `.github/workflows/` | `verify-contracts.yml` (ruff + 160 CPU tests, runs in GH Actions);
```

In `CLAUDE.md`, replace:

```markdown
         sakthai-sft-training/tests/    # 130 passed
```

with:

```markdown
         sakthai-sft-training/tests/    # 160 passed
```

- [ ] **Step 7: Verify no stale count remains**

Run:

```bash
grep -nE "169|149 Tests|130 passed|94 CPU" README.md CLAUDE.md
```

Expected: no output. Any hit is a count site that was missed.

- [ ] **Step 8: Verify the counts still match reality**

Run:

```bash
export UV_INDEX_URL=https://pypi.org/simple UV_DEFAULT_INDEX=https://pypi.org/simple
uv run --with datasets --with pytest --with pydantic --with httpx --with anyio \
  pytest test_browsergym_contract.py \
         sakthai-agentic-eval-train/tests/ \
         openenv-custom-training/tests/ \
         openenv-multi-catalog-training/tests/ \
         sakthai-sft-training/tests/ -q
```

Expected: `160 passed`. If this number is not 160, the documentation must be updated to the new value in the same commit, and the same must be true for the per-suite counts in step 3.

- [ ] **Step 9: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: correct the test counts to the measured values

The suite collects 160 tests (root 15, agentic-eval 30, custom 53,
multi-catalog 25, SFT 37), measured 2026-11-26. The repository stated
169, 149, 130, and 94 in five places, and no two agreed.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Published checkpoints distinguished from training targets

**Files:**
- Modify: `README.md:28`, `README.md:52`
- Modify: `docs/MODELS.md:15`, `docs/MODELS.md:150`
- Modify: `sakthai-agentic-eval-train/FINDINGS.md:170`, `:175`, `:177`, `:179`

**Interfaces:**
- Consumes: nothing.
- Produces: no site presents an unpublished model as published. The `TRAIN_PUSH_TO` targets are untouched.

- [ ] **Step 1: Update the README published-checkpoint bullet**

In `README.md`, replace:

```markdown
   - Produces final RL-aligned checkpoints: [`Nanthasit/sakthai-context-7b-tools-grpo`](https://huggingface.co/Nanthasit/sakthai-context-7b-tools-grpo).
```

with:

```markdown
   - Targets final RL-aligned checkpoints at [`Nanthasit/sakthai-context-7b-tools-grpo`](https://huggingface.co/Nanthasit/sakthai-context-7b-tools-grpo) — **not yet published**; the 40-step run was proof-of-signal only (see `docs/KNOWN_GAPS.yaml#grpo-proof-run-too-short`).
```

- [ ] **Step 2: Update the README capability matrix row**

In `README.md`, replace:

```markdown
| `sakthai-context-7b-tools-grpo` | 7B + GRPO | Production RL checkpoint aligned on OpenEnv tasks | In-loop eval | 🎯 **Final output** |
```

with:

```markdown
| `sakthai-context-7b-tools-grpo` | 7B + GRPO | RL checkpoint target — not yet published (40-step proof-of-signal only) | In-loop eval | 🎯 **Planned output** |
```

- [ ] **Step 3: Update the MODELS.md portfolio row**

In `docs/MODELS.md`, replace:

```markdown
| **`sakthai-context-7b-tools-grpo`** | 7B + GRPO | 7B | Post-RL Checkpoint | Production Aligned Checkpoint | Designated target repo for RL training rollouts on OpenEnv environments | [🤗 Hub Target](https://huggingface.co/Nanthasit/sakthai-context-7b-tools-grpo) |
```

with:

```markdown
| **`sakthai-context-7b-tools-grpo`** | 7B + GRPO | 7B | Post-RL Checkpoint | Planned Aligned Checkpoint — not yet published | Designated target repo for RL training rollouts on OpenEnv environments | [🤗 Target repo](https://huggingface.co/Nanthasit/sakthai-context-7b-tools-grpo) |
```

- [ ] **Step 4: Update the MODELS.md orchestration diagram**

In `docs/MODELS.md`, replace:

```text
[ Sak-Family-Agent ] ──orchestrates─┼── sakthai-context-7b-tools-grpo (Action / Tool Loop)
```

with:

```text
[ Sak-Family-Agent ] ──orchestrates─┼── sakthai-context-7b-tools-grpo (Action / Tool Loop, planned)
```

- [ ] **Step 5: Update the FINDINGS.md artifacts heading**

In `sakthai-agentic-eval-train/FINDINGS.md`, replace:

```markdown
## 5. Artifacts produced (all on the Hub)
```

with:

```markdown
## 5. Artifacts produced

Published to the Hub unless noted otherwise.
```

- [ ] **Step 6: Update the three FINDINGS.md artifact bullets**

In `sakthai-agentic-eval-train/FINDINGS.md`, replace:

```markdown
- **`Nanthasit/sakthai-context-0.5b-tools-sft`** — the SFT bootstrap checkpoint (real,
  useful; agentic 1/6, single-shot 77.8%).
```

with:

```markdown
- **`Nanthasit/sakthai-context-0.5b-tools-sft-v2`** — the SFT bootstrap checkpoint (real,
  useful; agentic 1/6, single-shot 77.8%). Republished under the `-v2` name; the original
  `-sft` repo is gone.
```

In `sakthai-agentic-eval-train/FINDINGS.md`, replace:

```markdown
- **`sakthai-context-0.5b-tools-sft-grpo`** — pushed but **byte-identical to the SFT
  model** (GRPO made zero updates); not useful, safe to delete.
```

with:

```markdown
- **`sakthai-context-0.5b-tools-sft-grpo`** — trained but **byte-identical to the SFT
  model** (GRPO made zero updates); not useful, safe to delete. Not published to the Hub.
```

In `sakthai-agentic-eval-train/FINDINGS.md`, replace:

```markdown
- **`sakthai-context-7b-tools-grpo`** — 40-step 7b GRPO (proof-of-signal only, saved
  fp32/bloated; supersede with the long run above).
```

with:

```markdown
- **`sakthai-context-7b-tools-grpo`** — 40-step 7b GRPO (proof-of-signal only, saved
  fp32/bloated; supersede with the long run above). Not published to the Hub.
```

- [ ] **Step 7: Verify the push targets are untouched**

Run:

```bash
grep -rn "TRAIN_PUSH_TO" --include="*.md" . | grep -v "^./.git"
```

Expected: five hits, unchanged — `CLAUDE.md`, `PLAN.md`, `README.md`, `sakthai-agentic-eval-train/FINDINGS.md`, and `sakthai-agentic-eval-train/README.md`.

- [ ] **Step 8: Verify against live Hub state**

Run:

```bash
for m in sakthai-context-7b-tools-grpo sakthai-context-0.5b-tools-sft sakthai-context-0.5b-tools-sft-v2; do
  printf "%-40s HTTP=%s\n" "$m" "$(curl -sS -o /dev/null -w '%{http_code}' --max-time 20 "https://huggingface.co/Nanthasit/$m")"
done
```

Expected: `401` for `sakthai-context-7b-tools-grpo`, `401` for `sakthai-context-0.5b-tools-sft`, `200` for `sakthai-context-0.5b-tools-sft-v2`. These are the values the corrected text now describes.

- [ ] **Step 9: Commit**

```bash
git add README.md docs/MODELS.md sakthai-agentic-eval-train/FINDINGS.md
git commit -m "docs: distinguish published checkpoints from training targets

The GRPO checkpoint and the original SFT bootstrap are not on the Hub;
both return 401 and the author publishes no -grpo model. Four sites
presented them as published artifacts. The push targets that point at
those repos are unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: CI description, local commands, and the changelog entry

**Files:**
- Modify: `CLAUDE.md:20-22`, `CLAUDE.md:89-100`
- Modify: `README.md:139-153`
- Modify: `CHANGELOG.md` (prepend a dated entry)

**Interfaces:**
- Consumes: the CI recipe from Task 3 and the verified counts from Task 4.
- Produces: documentation that describes the CI recipe and the local command accurately, plus a CHANGELOG entry recording every change in this plan.

- [ ] **Step 1: Update the CLAUDE.md CI toolchain description**

In `CLAUDE.md`, replace:

```markdown
**not** an install manifest for the pipeline — it is only the CPU dev/test toolchain
`verify-contracts.yml` installs (datasets, httpx, pydantic, pytest, ruff); its own header
explains why unifying the workspaces' pinsets there does not resolve.
```

with:

```markdown
**not** an install manifest for the pipeline — it is only the CPU dev/test toolchain
`verify-contracts.yml` installs (datasets, httpx, pydantic, pytest, ruff); its own header
explains why unifying the workspaces' pinsets there does not resolve. The GRPOConfig
contract check additionally installs CPU-only torch and the pinned `trl` into a separate
`.venv-verify/`, so the pytest environment keeps the `conftest.py` stubs.
```

- [ ] **Step 2: Update the CLAUDE.md local command block**

In `CLAUDE.md`, replace:

```bash
uv run --with datasets python3 verify_grpo_contract.py  # drives SimpleGuessEnv
```

with:

```bash
uv run --with datasets python3 verify_grpo_contract.py  # drives SimpleGuessEnv; skips the GRPOConfig check without trl
```

And in the same block, after the `verify_grpo_contract.py` line, add the full-dependency variant:

```bash
uv venv .venv-verify && uv pip install --python .venv-verify --torch-backend cpu torch && \
  uv pip install --python .venv-verify "trl==1.18.1" "transformers>=5.23.0" datasets && \
  .venv-verify/bin/python verify_grpo_contract.py  # includes the GRPOConfig check
```

- [ ] **Step 3: Update the README local command block**

In `README.md`, replace:

```bash
uv run --with datasets python3 verify_grpo_contract.py
```

with:

```bash
uv run --with datasets python3 verify_grpo_contract.py
```

and add immediately after the existing OpenEnv RL Contract Verification block:

```bash
The same check with the GRPOConfig section enabled (requires CPU torch and trl):

uv venv .venv-verify && uv pip install --python .venv-verify --torch-backend cpu torch && \
  uv pip install --python .venv-verify "trl==1.18.1" "transformers>=5.23.0" datasets && \
  .venv-verify/bin/python verify_grpo_contract.py
```

- [ ] **Step 4: Prepend the CHANGELOG entry**

In `CHANGELOG.md`, insert after the intro block (the `---` separator that follows the "Repo-level landed work, newest first" paragraph) and before the `## 2026-11-21` heading:

```markdown
## 2026-11-26

### Fixed

- **Echo Episode Termination** — `MultiCatalogEnv.echo_say` set
  `self.done = result.observation.done or True`, which is unconditionally `True`
  because Python's `or` yields the first truthy operand. The server's episode state
  was discarded and every echo episode ended after a single tool call. Now
  `self.done = result.observation.done`. The guarding test asserted `env.done is True`
  against a mock already reporting `done = True`, so it passed under both the buggy
  and the correct implementation; it now asserts the environment follows the server in
  the non-terminal case, with a terminal case kept alongside.
- **GRPOConfig Contract Check** — the fourth check in `verify_grpo_contract.py` never
  executed in CI, because CI installs only `datasets` and the check skips when `trl` is
  absent. Enabling it surfaced two further problems: `GRPOConfig` requires
  `use_cpu=True` on a CPU box, and `per_device_train_batch_size` must be divisible by
  `num_generations` (the script passed 1 against a default of 8). The check now
  constructs with `per_device_train_batch_size=4, num_generations=4` and explicitly
  asserts that `vllm_server_url` is rejected with `TypeError`.

### Changed

- **CI Contract Step** — `verify-contracts.yml` now installs CPU-only torch and the
  pinned `trl==1.18.1` into a dedicated `.venv-verify/` before running the contract
  script, so the GRPOConfig check actually asserts something. The pytest environment is
  unchanged, because `conftest.py` prefers the real `trl` package over its permissive
  stub. The job timeout moves from 10 to 15 minutes to cover the install.
- **Test Counts** — the suite collects 160 tests (root 15, agentic-eval 30, custom 53,
  multi-catalog 25, SFT 37), measured 2026-11-26. README.md and CLAUDE.md stated 169,
  149, 130, and 94 across eight sites, and no two agreed. The 169 figure in the
  2026-11-21 entry above was never accurate: that commit already collected 160 tests.
- **Published vs Targeted Checkpoints** — `Nanthasit/sakthai-context-7b-tools-grpo` and
  `Nanthasit/sakthai-context-0.5b-tools-sft` are not published (both return 401, and
  the author publishes no `-grpo` model). README.md, `docs/MODELS.md`, and FINDINGS.md
  now mark them as targets or as unpublished artifacts. The SFT bootstrap is recorded
  under its current `-v2` name. The `TRAIN_PUSH_TO` push targets are unchanged.

```

- [ ] **Step 5: Verify the CHANGELOG is newest-first**

Run `head -40 CHANGELOG.md`

Expected: the `## 2026-11-26` heading appears above `## 2026-11-21`, and the 2026-11-21 entry is unmodified.

- [ ] **Step 6: Run the complete verification suite**

Run:

```bash
export UV_INDEX_URL=https://pypi.org/simple UV_DEFAULT_INDEX=https://pypi.org/simple
uv run --with ruff ruff check .
uv run --with datasets --with pytest --with pydantic --with httpx --with anyio \
  pytest test_browsergym_contract.py \
         sakthai-agentic-eval-train/tests/ \
         openenv-custom-training/tests/ \
         openenv-multi-catalog-training/tests/ \
         sakthai-sft-training/tests/ -q
.venv-verify/bin/python verify_grpo_contract.py
git diff --check
```

Expected: `All checks passed!`, `160 passed`, `✨ All OpenEnv RL Contract Tests Passed Successfully!`, and no output from `git diff --check`.

- [ ] **Step 7: Commit**

```bash
git add CLAUDE.md README.md CHANGELOG.md
git commit -m "docs: update the CI recipe and record the Track A changes

CLAUDE.md and README.md now describe the CI recipe that installs CPU
torch and pinned trl, and the CHANGELOG records the defect fix, the
contract-check repair, the count corrections, and the published-versus-
targeted distinction.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final Verification

Run once after all six tasks:

```bash
export UV_INDEX_URL=https://pypi.org/simple UV_DEFAULT_INDEX=https://pypi.org/simple
cd /workspace/repos/openenv-rl-training
uv run --with ruff ruff check .
uv run --with datasets --with pytest --with pydantic --with httpx --with anyio \
  pytest test_browsergym_contract.py \
         sakthai-agentic-eval-train/tests/ \
         openenv-custom-training/tests/ \
         openenv-multi-catalog-training/tests/ \
         sakthai-sft-training/tests/ -q
.venv-verify/bin/python verify_grpo_contract.py
grep -nE "169|149 Tests|130 passed|94 CPU" README.md CLAUDE.md && echo "STALE COUNT FOUND" || echo "no stale counts"
git status --porcelain
git log --oneline main..HEAD
```

Expected, in order: `All checks passed!`, `160 passed`, the contract script's success line, `no stale counts`, an empty `git status --porcelain`, and nine commits ahead of `main` — the design spec, this plan, and seven implementation commits (the echo fix, the echo test, the contract-check repair, the CI wiring, the count corrections, the published-versus-targeted corrections, and the CI-recipe changelog entry).

The branch remains local and is not pushed.
