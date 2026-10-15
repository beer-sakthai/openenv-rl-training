# CHANGELOG

Repo-level landed work, newest first. Empirical findings stay in
[`sakthai-agentic-eval-train/FINDINGS.md`](sakthai-agentic-eval-train/FINDINGS.md);
open items stay in [`PLAN.md`](PLAN.md) and [`docs/KNOWN_GAPS.yaml`](docs/KNOWN_GAPS.yaml).
When a gap closes, delete its PLAN.md / KNOWN_GAPS.yaml entry and add a line here.

---

## 2026-10-15

### Changed

- **`README.md` status-bar badges** — removed 5 that structurally cannot be green on
  main: `auto-merge`, `dependency-review`, `summary` are `pull_request`-only so they
  never run on `main` (badge reads "no status", grey, looks failed); `mcp-bench` and
  `train` are `workflow_dispatch`-only and their only historical runs failed on HF Jobs
  402 before `hf-jobs-submit.sh` existed (badge sticks red forever until re-dispatched,
  which stays blocked on prepaid credit — see `KNOWN_GAPS.yaml#hf-jobs-402`). The 9
  remaining badges (verify-contracts, codeql, ossar, hf-no-cost-checks, stale,
  auto-update-prs, eval, lighteval, monitor) do run on `main` and reflect real health.
- **`CLAUDE.md § Known open items`** — HF Jobs 402 bullet trimmed from a 9-line
  paragraph to one sentence pointing at
  [`.github/scripts/hf-jobs-submit.sh`](.github/scripts/hf-jobs-submit.sh). The script
  header already carries the full "report 402 as a skip" rationale.
- **`sakthai-sft-training/push-v9-comprehensive.py`** — module docstring now flags the
  script as DEPRECATED (never run; `Nanthasit/sakthai-combined-v9` was never created on
  the Hub) and points at `docs/KNOWN_GAPS.yaml#dataset-v8-missing`. Script contents
  preserved intact as a record of what the v9 build intended.

## 2026-10-14

### Added

- **`CHANGELOG.md`** — this file. Landed work moves out of `PLAN.md` into here, so each
  file has one job. Covers the four landed sweeps the repo has accumulated.
- **`.github/CODEOWNERS`** — one-liner `* @beer-sakthai` routing every path to the
  maintainer so review-routing is explicit for the GitHub review bot and the Claude
  Approvals integration. (PR [#178](https://github.com/beer-sakthai/openenv-rl-training/pull/178))

### Changed

- **`PLAN.md`** trimmed from 91 → 61 lines. Open bullets regrouped into
  `hygiene / lint`, `infra / dependencies`, `real runs`, `monitoring`; every bullet
  cross-references the relevant `KNOWN_GAPS.yaml` anchor. (PR #178)
- **`CLAUDE.md` Layout table** — new `CHANGELOG.md` row; `PLAN.md` row updated to
  describe its new "open work only" role. (PR #178)
- CodeQL action bumped from 4.38.2 to 4.38.3 for both `init` and `analyze`
  (PRs [#174](https://github.com/beer-sakthai/openenv-rl-training/pull/174) and
  [#176](https://github.com/beer-sakthai/openenv-rl-training/pull/176)); the two
  must land together or CodeQL fails on the version mismatch.

## 2026-10-10

### Added

- **`docs/KNOWN_GAPS.yaml`** — parseable index of the 10 scattered "Known open items"
  from `CLAUDE.md` + `PLAN.md`, with `location` / `impact` / `workaround` / `fix` /
  `blocker` / `verified_on` fields per entry. Prose in `CLAUDE.md` stays authoritative;
  this file is the machine-readable mirror. (PR [#172](https://github.com/beer-sakthai/openenv-rl-training/pull/172))
- **`sakthai-sft-training/README.md`** benchmark table — new **GRPO target?** column.
  `0.5b-tools` ❌ marked SFT-only (settles the "retire 0.5b as GRPO target" follow-up
  from PLAN.md); `7b-tools` ✅ marked the only viable GRPO target. (PR #172)
- **README.md** status bar — new `📋 Known gaps — 10 tracked` shield linking to
  `docs/KNOWN_GAPS.yaml`, plus a `📋 Known-gaps index` row in the live-status table.
  (PR [#173](https://github.com/beer-sakthai/openenv-rl-training/pull/173))

### Changed

- **`CLAUDE.md § Known open items`** — v8-load wording corrected. The
  `train-sakthai-1.5b-v2.py:88` load is wrapped in `try/except` that prints
  `v8 unavailable:` and continues, so `train.yml` silently drops the v8 augmentation
  shard rather than failing. Readers are pointed at `v10` as the latest existing
  revision for a repoint fix. (PR #172)

## 2026-10-09

### Changed

- **Dependency pass + Dependabot CI** (PR #171 and others). 8 Dependabot PRs resolved
  (6 merged, 2 superseded) — bumps now on `main`: `openenv` 0.7.0, `transformers`
  ≥5.19.0, `vllm` ≥0.31.0, `datasets` ≥5.1.0 in `openenv-custom-training`,
  `step-security/harden-runner` 2.22.0.
- **`hf-no-cost-checks.yml`** now soft-skips (warn + exit 0) when `HF_TOKEN` is
  withheld, as it is from Dependabot and fork PRs by GitHub design; push:main and the
  weekly cron still catch real token drift.

## 2026-09-16 (repo-health pass)

### Fixed

- **`sakthai-sft-training/push-all-to-hub.py`** — unterminated module docstring closed;
  the only file `compileall` had rejected.
- **`train.py --env browsergym`** — reward now reads `env.reward` off the instance via a
  proper wrapper (was always 0.0), and the dataset is conversational (was a bare string).
  Both contract tests had been asserting the bugs; they assert the correct contract now.
- **`a2a_agent/agent_executor.py`** — 13 duplicated methods from a bad merge removed
  (423 → 340 lines); live behaviour proven unchanged.
- **Two NameErrors in eval scripts** (`del m`, missing `import collections`) fixed; both
  had fired partway through a paid GPU job.
- **27 workspace tests no CI job ran** — root `conftest.py` now stubs `openenv` when the
  real package is absent, so the workspace suites collect on a CPU box; `verify-contracts.yml`
  runs 38 tests (plus ruff) on every PR.

### Changed

- **`monitor.yml`** no longer fails every Monday by design (prior scheduled run
  pointed at a dataset the token couldn't publish to).
- **`train.yml`** — nonexistent `hf-jobs-h200` runner label dropped; now runs on
  `ubuntu-latest` and dispatches `hf jobs uv run` like the other HF-Jobs workflows.
- **`mcp-bench.yml`** — now installs `hf` before calling it.
- **`manual.yml`** / **`stale.yml`** — unmodified GitHub starter templates deleted
  (`manual.yml`) or trimmed; `label.yml` deleted (fetches config from base on
  `pull_request_target`, so it fails on every first-time PR by construction).

## 2026-08-21 (consolidation PR)

Full analysis in [`docs/CONSOLIDATION.md`](docs/CONSOLIDATION.md).

### Added

- **`sakthai-sft-training/`** — the former `beer-sakthai/SakThai-Training` folded in
  as the SFT half: QLoRA training scripts (0.5B / 1.5B / 7B), 10-cycle
  data-augmentation loop (`cycle-100-v2..v10.py`), cross-model + MCP-Bench + lighteval
  evaluators, `sakthai-cycle-bench/` (155-row BFCL harness), ops tooling, SFT Colab
  notebook, publishing helpers, gap-fill data.
- **`.github/workflows/`** seeded from SakThai-Training (`train`, `eval`, `lighteval`,
  `mcp-bench`, `monitor`); paths rewritten to `sakthai-sft-training/…`.
- **`.github/workflows/verify-contracts.yml`** — runs `verify_grpo_contract.py` and
  `test_browsergym_contract.py` on every PR.
- **`.opencode/`** (25 commands + 35 skills) lifted verbatim; no path rewrites needed.
- **`SECURITY.md`** and **`docs/HF_HUB_IMPROVEMENTS.md`** lifted; `AGENTS.md` folded
  into `CLAUDE.md § "The SFT half"`.
- **`README.md`** and **`CLAUDE.md`** rewritten to cover both halves of the pipeline.
- **`.eval_results/`** benchmark YAMLs preserved from the retired repo and moved to
  `sakthai-sft-training/sakthai-cycle-bench/eval_results/`.

### Fixed

- **`openenv-multi-catalog-training/train_multi_env.py:main()`** —
  `vllm_server_url` → `vllm_server_host` + `vllm_server_port` (the former does not
  exist in `GRPOConfig` and crashes at construction).
- **`openenv-custom-training/train.py --env browsergym`** — `browsergym_env` dep
  added to the PEP 723 header and `requirements.txt`.

### Changed

- **`train.yml`** set to `workflow_dispatch`-only to avoid unintentional paid GPU spend.
- Upstream sync: 185 commits pulled (`main` → `96580a1`), including root
  `requirements.txt` and unit tests under `sakthai-agentic-eval-train/tests/` and
  `openenv-custom-training/tests/`.

### Removed

- **`beer-sakthai/SakThai-Training` on GitHub** — `gh repo delete` on 2026-08-21 after
  content-parity sweep (0 gaps). Local checkout archived to
  `/home/beern/archive/SakThai-Training` with full git history.
