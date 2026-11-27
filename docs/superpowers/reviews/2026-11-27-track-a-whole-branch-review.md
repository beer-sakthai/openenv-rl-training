# Whole-Branch Review — Track A Verification Truthfulness Remediation

**Date:** 2026-11-27 · **Reviewed range:** `a65fb23..09e9201` (10 commits, pre-rebase)
**Post-review state:** branch rebased onto `main` (`0611da1`, PR #249); see
[§6 Post-review rebase](#6-post-review-rebase).

**Verdict: With fixes.** The code half of the branch is correct and independently
verified. The remaining problems are documentary or defensive, not behavioural.

---

## 1. Strengths

### 1.1 The headline fix is real, minimal, and independently verified

`openenv-multi-catalog-training/train_multi_env.py:226`

```diff
-        self.done = result.observation.done or True
+        self.done = result.observation.done
```

Python's `or` yields the first truthy operand, so the old expression was
unconditionally `True` and the server's episode state was discarded.

The spec's mutation test was reproduced independently, not taken on trust: in a
scratch copy of the tree, restoring the buggy line makes the strengthened test fail
with exactly `assert True is False`; with the fix it passes. The sibling one-shot
tools (`coding_run:258`, `repl_finish:340`) are untouched and still deliberate.

### 1.2 The strengthened test is genuinely discriminating, and the count holds

`openenv-multi-catalog-training/tests/test_multi_catalog_env.py:213-239` flips the
echo case to `observation.done = False` -> `assert env.done is False`, and keeps the
terminal case at `:232-239` asserting `env.done is True`.

The old test asserted `env.done is True` against a mock already reporting
`done = True` — non-discriminating under both implementations. The case is
extended, never duplicated.

### 1.3 Every count claim is measured-true

The CI invocation was run against the branch:

| Suite | Measured |
|---|---|
| root contract (`test_browsergym_contract.py`) | 15 |
| `sakthai-agentic-eval-train/tests/` | 30 |
| `openenv-custom-training/tests/` | 53 (59 after PR #249) |
| `openenv-multi-catalog-training/tests/` | 25 (30 after PR #249) |
| `sakthai-sft-training/tests/` | 37 |
| **total** | **160** (171 after PR #249) |

The "six sites" claim is right: the base had exactly six stale-count sites
(`README.md:9/97/243`, `CLAUDE.md:31/76/97`).

### 1.4 The publication claims are correct against live Hub state

`Nanthasit/sakthai-context-7b-tools-grpo` -> 401, `sakthai-context-0.5b-tools-sft`
-> 401, `sakthai-context-0.5b-tools-sft-v2` -> 200, `sakthai-context-0.5b-tools-sft-grpo`
-> 401. The Hub API lists exactly **18** models for `Nanthasit` and **none** containing
"grpo" — so the CHANGELOG's strongest claim holds.

Every remaining `-grpo` mention (`PLAN.md:35`, `docs/MODELS.md:45`,
`docs/CONSOLIDATION.md:128`, all `TRAIN_PUSH_TO=` sites) is a legitimate push target.

### 1.5 The CHANGELOG's hardest historical claim is true

> "The 169 figure in the 2026-11-21 entry was never accurate: that commit already
> collected 160 tests."

Verified independently: the tree at `a91f95d` was extracted with `git archive` and
measured **160 collected** (159 passed + 1 pre-existing environment-dependent failure).

### 1.6 The contract check is now a genuine guard

Simulating a TRL that accepts `vllm_server_url` makes the check print
`❌ GRPOConfig accepted vllm_server_url` and exit **1**. With the real pinned TRL it
exits 0 and all four sections pass.

The divisibility rationale is real: the old `per_device_train_batch_size=1` against
`num_generations=8` raises
`ValueError: generation_batch_size (1) must be divisible by num_generations (8)`.

### 1.7 The CI isolation rationale is exactly as described

`conftest.py:408-427` prefers the real `trl` and falls back to a permissive stub only
on `ImportError` — so the dedicated venv is genuinely required, and the pytest step's
non-change is correct, not superstition.

`.github/workflows/verify-contracts.yml` parses as valid YAML;
`uv venv` + `uv pip install --python .venv-verify --torch-backend cpu torch` was
verified working; `ruff check .` -> `All checks passed!`; 15 minutes is proportionate
against the spec's measured 2m26s CPU install.

### 1.8 Hygiene is good

`.gitignore` gains `.venv-verify/` (keeps a 1.2 GB venv out of git). Ten commits, one
logical change each, repo-scoped
`beer-sakthai <293853443+beer-sakthai@users.noreply.github.com>` identity and the
`Co-Authored-By` trailer on every one; `main`/`origin/main` untouched at the time of
review; no remote branch.

---

## 2. Issues

### Critical (Must Fix)

None. Nothing here produces incorrect runtime behaviour or a false claim that reaches
a user-facing doc or a CI verdict.

### Important (Should Fix)

**I1 — The two documents this branch ships assert a `transformers` pin that the
requirements file contradicts.**
`docs/superpowers/specs/2026-11-26-track-a-verification-remediation-design.md:211`
and `:217`; `docs/superpowers/plans/2026-11-26-track-a-verification-remediation.md:15`,
`:348`, `:388`, `:792`, `:816` all said `transformers>=5.2.0` and that it was
"matching the same requirements file". `openenv-custom-training/requirements.txt:12`
says `transformers>=5.23.0`. The `5.2.0` figure was transcribed from the explanatory
comment at `requirements.txt:10` (TRL's *floor*, not the repo's pin).
**Status: fixed on 2026-11-27** — pin corrected to `>=5.23.0` in both documents, with
the distinction between TRL's floor and the repo's pin stated explicitly.

**I2 — `verify_grpo_contract.py:190-191` silently swallows the `ImportError`,
recreating the exact no-op this branch removed.**
If a future `transformers` major breaks `trl==1.18.1`'s import chain,
`from trl import GRPOConfig` raises `ImportError` -> the script prints
"⚠️ TRL package not installed… skipping" -> exits **0** -> CI is green with the guard
silently absent. That is the failure mode this branch exists to eliminate.
*Fix:* add an env-gated strict mode (e.g. `VERIFY_REQUIRE_TRL=1` -> `_fail` instead of
skip) and set it in the workflow; or assert the step's output contains the GRPOConfig
section.

**I3 — `README.md:88` — "**Capped Steps**: All environments enforce step limits
(`MAX_ATTEMPTS`, `STEP_LIMIT`)" is false for the env this branch modified.**
`MAX_ATTEMPTS`/`STEP_LIMIT` exist only under `openenv-custom-training/` and the
skill-scaffolding references; `openenv-multi-catalog-training/train_multi_env.py` has
**no step counter at all**. The claim sits under the lead-in "Every environment in this
repo honours the TRL `environment_factory` contract" (`README.md:83`) — in a file this
branch already edited, about the exact env whose termination semantics it just changed.
*Fix:* scope the bullet ("the Tier A/B custom environments enforce
`MAX_ATTEMPTS`/`STEP_LIMIT`; the multi-catalog env delegates termination to the
upstream catalog servers").

**I4 — The echo fix's actual runtime consequence is undocumented, and it is the
opposite of what the spec implies.**
Upstream OpenEnv's echo environment returns `done=False` **always** (HF OpenEnv docs:
"`done` (bool) - Always `False` for the echo environment"), and this repo runs
`registry.hf.space/openenv-echo-env:latest`
(`openenv-multi-catalog-training/run_servers.sh:41`). So after the fix, `echo_say`
never terminates the episode from the server.

The fix is still correct and valuable — it lets the model retry a wrong echo instead
of being locked into one shot, which is the strongest argument for it — but the spec's
Risk row said only "It restores intended behaviour." A reader would not learn that the
echo episode now has *no environment-side termination*, relying instead on the model
stopping or TRL's `max_completion_length` (`max_tool_calling_iterations` defaults to
unlimited — `trl/trainer/grpo_config.py:925`, `grpo_trainer.py:843`).
*Fix:* one sentence in the spec §3.1/§7 and the CHANGELOG entry.

**I5 — `sakthai-agentic-eval-train/FINDINGS.md:178-179` overclaims what the Hub
evidence supports.**
"Republished under the `-v2` name; the original `-sft` repo is gone" asserts a
republication *event* and a *deletion*. Hugging Face returns 401 for both nonexistent
**and** private repos, so the evidence supports only "the `-sft` name is not publicly
accessible."
*Fix:* "the original `-sft` name is not publicly accessible (HTTP 401); the checkpoint
is published under `-v2`."

**I6 — `sakthai-agentic-eval-train/FINDINGS.md:77` vs `:177` — a naming inconsistency
this branch introduced.**
`:77` reads "After SFT (0.5b-tools-sft)"; `:177` now reads
`sakthai-context-0.5b-tools-sft-v2`. Before `c7b011a` these agreed (both said `-sft`);
the branch changed one and left the other.
*Fix:* change `:77` to `(0.5b-tools-sft-v2)`.

**I7 — `verify_grpo_contract.py:153-193` never records which TRL it validated.**
The CI log cannot evidence the plan's central guarantee ("a guard that validates a
different TRL than the training code will use is worthless").
*Fix:* one line printing `trl.__version__` (and `transformers.__version__`) in the
section.

**I8 — `verify_grpo_contract.py:154` — the section title over-promises.**
It prints "Verifying GRPOConfig field compatibility" but exercises only
`vllm_server_host`/`_port` plus the `vllm_server_url` negative.
`openenv-custom-training/train.py:373-384` passes seven unconditional fields
(`use_vllm`, `vllm_mode`, `max_completion_length`, `num_generations`,
`gradient_accumulation_steps`, `chat_template_kwargs`, `log_completions`) plus the two
vLLM server fields — the same "nonexistent field crashes at construction" class could
strike any of the seven unguarded ones.
*Fix:* construct with the training code's field set, or retitle to "GRPOConfig vLLM
server field compatibility".

**I9 — `verify_grpo_contract.py:26-27` names the wrong field for the divisibility
rule.**
TRL 1.18.1's rule (`trl/trainer/grpo_config.py:1172`) is
`generation_batch_size % num_generations != 0`; `generation_batch_size` defaults to
`per_device_train_batch_size * world_size * gradient_accumulation_steps` (`:1139-1141`),
so the two coincide **only under the defaults**. The docstring (and the CHANGELOG's
echo of it) states the rule as being on `per_device_train_batch_size`, which will
mislead anyone who sets `generation_batch_size` or `steps_per_generation`.
*Fix:* name `generation_batch_size` and note the default equivalence.

**I10 — `.github/workflows/verify-contracts.yml:42-44` — the comment gives only one of
the two reasons for the dedicated venv.**
It explains the pytest-stub hazard but omits that `uv run --with` cannot select the CPU
wheel — a fact the plan's Global Constraints record and the spec measured
(`spec §3.3`: 5m20s CUDA vs 2m26s CPU, plus `nvidia-cublas`/`cudnn`/`nccl`). A
maintainer "simplifying" this step to `uv run --with trl==1.18.1` silently pulls
multi-GB CUDA wheels and likely blows the 15-minute timeout.
*Fix:* one extra comment line.

**I11 — `docs/MODELS.md:31` — the `my-embedding-gemma` label is stale.**
The URL 307-redirects to `Nanthasit/Sak-Gem` (verified), and the Hub list shows the
model under its new name only. The link still resolves, so this is cosmetic rather
than a false claim.
*Fix:* relabel the row `Sak-Gem` (keeping the 18-model count intact).

### Minor (Nice to Have)

**M1 — `verify_grpo_contract.py:189` prints "✅ GRPOConfig vLLM parameters
compatibility passed." unconditionally, including after `_fail()` at `:185`.** A
failing run logs `❌ GRPOConfig accepted vllm_server_url…` immediately followed by
`✅ … passed.` The exit code stays 1 so CI fails correctly, but the log is
self-contradictory. *Fix:* move the success print into an `else:` branch, or drop it —
`_section` already prints per-section verdicts for sections 1-3.

**M2 — `openenv-multi-catalog-training/tests/test_multi_catalog_env.py:213` — dead
`env = MultiCatalogEnv()`** immediately shadowed by the fresh instance at `:216`.
*Fix:* delete line 213.

**M3 — `openenv-multi-catalog-training/a2a_agent/agent_executor.py:222` —
`_step_echo` sets `episode.done = True` unconditionally** while `_step_sudoku` at
`:233` mirrors the server. With the upstream echo env always returning `done=False`,
the A2A agent cannot retry a failed echo. Its test (`test_a2a_agent.py:122`) uses
`MagicMock(echoed_message=…)`, whose `.done` is a truthy MagicMock — the same
non-discriminating pattern the branch just fixed in the training env. Out of the
plan's scope, and the A2A path may legitimately be one-exchange-per-task, but it is
the same defect family. *Recommended as a follow-up PR, not a merge blocker.*

**M4 — `README.md:52`** — the "GRPO Target?" cell answers "🎯 **Planned output**",
which describes status while the other rows answer the column ("❌ SFT-only",
"⚠️ Not targeted", "✅ Primary target"). The Role column already carries "not yet
published", so this is redundant rather than wrong.

**M5 — `docs/MODELS.md:15`** — the Format cell still reads "Post-RL Checkpoint" for a
model that does not exist; the Specialty cell disambiguates it. Parallel readings
exist elsewhere in the file.

---

## 3. Deferred-finding triage

| # | Site | Verdict | Reason |
|---|---|---|---|
| 1 | `test_multi_catalog_env.py:213` dead binding | acceptable to defer | Cosmetic dead assignment; zero behavioural effect. |
| 2 | `a2a_agent/agent_executor.py:222` unconditional done | should fix (follow-up, not a blocker) | Same defect family as the fixed `train_multi_env.py:226`, and its test is non-discriminating; but it is a separate tool-serving path outside the plan's scope. |
| 3 | `verify_grpo_contract.py:185-189` "✅ passed" after `_fail` | should fix | A false success line in the CI log; one-line `else:`. Exit code is correct. |
| 4 | `verify_grpo_contract.py:186` `except TypeError: pass` | acceptable to defer | Structurally mitigated: the outer construction at `:162-170` already proves the five shared kwargs are valid, so the inner `TypeError` can only come from `vllm_server_url` itself. |
| 5 | `verify_grpo_contract.py:162-170` vs `:177-184` kwarg duplication | acceptable to defer | Adjacent 5-line duplication; low drift risk since the outer block runs first and would fail loudly. A shared `common = {...}` dict is a 3-line cleanup. |
| 6 | `verify_grpo_contract.py:15-17,154` no TRL version recorded | should fix | The CI log cannot evidence the plan's central "validates the training code's TRL" guarantee; one line to print. |
| 7 | Section title over-promises vs `train.py:373-384` | should fix | The title claims "field compatibility" while only two of ~nine fields the training code passes are exercised; the same crash class could strike an unguarded field. |
| 8 | `verify_grpo_contract.py:26-27` divisibility rule on wrong field | should fix | TRL's rule is on `generation_batch_size` (`grpo_config.py:1172`); the docstring and the CHANGELOG echo of it name `per_device_train_batch_size`. True only under defaults. |
| 9 | `verify-contracts.yml:49` unbounded transformers + `:190-191` swallows ImportError | should fix | The one path by which the branch's restored guard silently reverts to a no-op. |
| 10 | `verify-contracts.yml:42-44` comment omits the CUDA-wheel reason | should fix | The Global Constraints and the spec both record it; omitting it invites a future "simplification" to `uv run --with` that silently pulls multi-GB CUDA wheels. |
| 11 | `verify-contracts.yml:47` `uv venv` resolves from PATH | acceptable to defer | Verified benign: uv walks PATH directories in order and takes the first interpreter found in each; `actions/setup-python` prepends its tool-cache dir (containing `python3`/`python3.11`), so 3.11 wins before `/usr/bin` is reached. Tested with a versioned `python3.12` in a later PATH dir. |
| 12 | `docs/CONSOLIDATION.md:69` "(3 passed)" | should fix — **yes, put it in the fix wave** | Live, undated measurement parenthetical; the root contract file collects 15. Materially different from `CLAUDE.md:457`/`:423`, which sit inside explicitly dated "**Resolved <date>**" ledger entries where rewriting would falsify a dated record. `docs/CONSOLIDATION.md`'s header date (2026-08-21) dates the *consolidation*, not each Phase-4 checklist measurement. |
| 13 | `CLAUDE.md:457` "57 tests plus ruff" | leave (ruled) | Dated ledger entry; 57 was true on 2026-09-16. |
| 14 | `CLAUDE.md:423` "(9 tests passing)" | leave (ruled) | Dated ledger entry; 9 is the current per-file count for `test_a2a_agent.py`. |
| 15 | `FINDINGS.md:177-179` republication/deletion claim | should fix | 401 cannot distinguish deleted from private; the text asserts two events the evidence does not support. |
| 16 | `docs/MODELS.md:15` Format cell "Post-RL Checkpoint" | acceptable to defer | The Specialty cell disambiguates; parallel phrasing exists elsewhere in the file. |
| 17 | `README.md:52` "🎯 Planned output" in "GRPO Target?" | acceptable to defer | Describes status rather than answering the column, but the Role column carries the real answer ("not yet published"). |
| 18 | `FINDINGS.md:77` vs `:177` naming | should fix | **Branch-introduced**: the two agreed before `c7b011a`; the branch changed one and left the other. |
| 19 | `docs/MODELS.md:31` `my-embedding-gemma` 307 -> `Sak-Gem` | should fix | Label is stale relative to the rename (verified via redirect + Hub API); one-word fix, same class as the other doc corrections. |
| 20 | `FINDINGS.md:184-186` scratchpad scripts under the Hub lead-in | acceptable to defer | Pre-existing and *improved* by this branch (the lead-in now says "unless noted otherwise" and the bullet already carries "(scratchpad)"); the scripts are tracked in-repo at `sakthai-agentic-eval-train/scripts/`. |
| 21 | `README.md:28` / `docs/MODELS.md:15` link to 401 repos | acceptable to defer | Inherent to naming a push destination; both sites now label them "not yet published" / "Target repo". |

### Findings added by this review (not on the deferred list)

- **I1** — the shipped spec and plan assert `transformers>=5.2.0` "matching the same
  requirements file", which `requirements.txt:12` contradicts; 7 sites across the two
  new documents. **Fixed 2026-11-27.**
- **I3** — `README.md:88`'s "All environments enforce step limits" is false for
  `MultiCatalogEnv`.
- **I4** — the echo fix's consequence (upstream echo env always returns `done=False`) is
  undocumented.
- **I5** — `FINDINGS.md:178-179` asserts a republication event and a deletion from
  401-vs-200 evidence only.
- **I6** — `FINDINGS.md:77`/`:177` naming inconsistency.
- **M2** — the dead `env = MultiCatalogEnv()` at `test_multi_catalog_env.py:213`.

---

## 4. Fix-wave scope confirmation

Both items already routed to the fix wave are correctly scoped and complete:

- **`README.md:147-153`** — every bash fence in `README.md` and `CLAUDE.md` was
  validated: this is the **only** invalid one (`bash -n` exits 2, `syntax error near
  unexpected token '('`); the other 14 exit 0. One block, one file; the repo's own
  convention for prose in a bash fence is a `#` comment (`README.md:115-127`).
  *Scope adequate.*
- **`CHANGELOG.md:24-26`** — confirmed against `git show a65fb23:verify_grpo_contract.py`:
  `use_cpu=True` was **already present** at the base commit, so the "surfaced two
  further problems" framing mis-attributes it. The divisibility half is true
  (verified). *Scope adequate:* rewrite to "Enabling it surfaced that
  `per_device_train_batch_size` must be divisible by `num_generations` (the script
  passed 1 against a default of 8)."

---

## 5. Recommendations

1. **Make the contract check self-certifying.** Print `trl.__version__` /
   `transformers.__version__`, and gate the `ImportError` skip behind an env var so CI
   can demand the check actually ran. Closes deferred #6 and #9 together and makes the
   CI log provable rather than assumed.
2. **One shared kwargs dict** in `verify_grpo_contract.py` (`common = {...}`), then
   construct with the training code's full field set — closes deferred #5, #7, and
   part of #4 in a single edit.
3. **Anchor the epoch of the test counts in the artifacts themselves.** The counts are
   dated in the CHANGELOG but the README/CLAUDE.md values are undated and will drift
   again. Consider a CI step that asserts the collected count equals the number in the
   README — that alone would have caught the PR #249 drift.
4. **Make the fix wave a sweep, not three patches.** Adding I1 (done), I3, I5, I6 and
   deferred #12 to it makes it a coherent "remaining false claims" pass matching the
   branch's thesis.
5. **Process note:** the two per-task review misses that survived to the whole-branch
   level — the false `use_cpu` attribution and the `transformers` pin slip — were both
   transcription errors of a nearby explanatory comment (`requirements.txt:10`). Worth
   a rule: when a spec quotes a pin from a requirements file, quote the pin line, not
   the comment above it, and re-verify quoted pins against the file before merging.
6. **Consider a `docs/MODELS.md` refresh** as separate work: the 18-model matrix's
   `my-embedding-gemma` -> `Sak-Gem` rename is the only live drift found, and the
   model list is otherwise exactly right.

---

## 6. Post-review rebase

After this review was written, PR #249 merged to `main` (`0611da1`), adding 11 tests
(`openenv-custom-training` 53 -> 59, `openenv-multi-catalog-training` 25 -> 30). The
branch was rebased onto that `main`; the rebase was conflict-free (the only overlapping
file, `openenv-multi-catalog-training/tests/test_multi_catalog_env.py`, was edited in
disjoint regions — line 12 and line 320+ on `main` versus lines 213-239 here).

Consequences handled on 2026-11-27:

- The live counts in `README.md`, `CLAUDE.md` and this branch's CHANGELOG entry were
  re-measured and updated to **171** (root 15, agentic-eval 30, custom 59,
  multi-catalog 30, SFT 37).
- The spec and plan each carry a dated amendment note recording the drift, so neither
  document makes a false claim.
- The rebased tree was re-verified: `171 passed`, `ruff check .` clean, the contract
  script green in the pinned venv, and the discriminating echo test confirmed still
  failing against the pre-fix implementation.

The reviewed range `a65fb23..09e9201` is preserved in the commit history of this
branch's pre-rebase commits and in the review package at
`.superpowers/sdd/2026-11-26-track-a-verification-remediation/`.
