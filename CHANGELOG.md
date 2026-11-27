# CHANGELOG

Repo-level landed work, newest first. Empirical findings stay in
[`sakthai-agentic-eval-train/FINDINGS.md`](sakthai-agentic-eval-train/FINDINGS.md);
open items stay in [`PLAN.md`](PLAN.md) and [`docs/KNOWN_GAPS.yaml`](docs/KNOWN_GAPS.yaml).
When a gap closes, delete its PLAN.md / KNOWN_GAPS.yaml entry and add a line here.

---

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
- **Test Counts** — the suite collects 171 tests (root 15, agentic-eval 30, custom 59,
  multi-catalog 30, SFT 37), measured 2026-11-27. README.md and CLAUDE.md stated 169,
  149, 130, and 94 across six sites, and no two agreed. The 169 figure in the
  2026-11-21 entry above was never accurate: that commit already collected 160 tests.
  PR #249 merged to `main` after this branch was cut and added 11 more (custom 53 → 59,
  multi-catalog 25 → 30), so the counts above are the post-merge values.
- **Published vs Targeted Checkpoints** — `Nanthasit/sakthai-context-7b-tools-grpo` and
  `Nanthasit/sakthai-context-0.5b-tools-sft` are not published (both return 401, and
  the author publishes no `-grpo` model). README.md, `docs/MODELS.md`, and FINDINGS.md
  now mark them as targets or as unpublished artifacts. The SFT bootstrap is recorded
  under its current `-v2` name. The `TRAIN_PUSH_TO` push targets are unchanged.

## 2026-11-21

### Added

- **Lifecycle & Resource Cleanup** — added deterministic `__del__` teardown hooks and multi-environment session switching cleanup (`_close_all`, per-task release) across `MultiCatalogEnv`, `_BrowserGymTaskEnv`, `AgentToolEnv`, `SandboxEnvironment`, and `MCPTaskEnv` to eliminate session and tempfile leaks across long GRPO episodes.
- **Robust `<tool_call>` Markdown Parsing** — updated `TOOL_CALL_PATTERN` and `parse_tool_calls` in both `pipeline_common.py` and `eval-cross-model.py` to seamlessly strip markdown code blocks (````json ... ````) inside tool call tags emitted by LLMs.
- **Explicit UTF-8 Encoding in Sandbox Server** — enforced explicit `encoding="utf-8"` on all `read_text` and `write_text` calls in `sandbox_env.py` for safe Thai unicode handling.
- **Test Suite Expansion to 169 Tests** — added unit and contract regression tests for sandbox UTF-8 operations and cleanup, agent tools teardown hooks, multi-catalog client lifecycle and switching, BrowserGym `__del__` session release, and markdown-fenced tool call parsing across all 5 workspace suites (169 passing tests).
- **`docs/MODELS.md` 18-Model Portfolio Reference Synchronization** — updated master catalog to reflect all 18 models on the Hugging Face Hub under `Nanthasit` (incorporating `sakthai-context-0.5b-merged`, `sakthai-context-1.5b-merged`, `sakthai-context-1.5b-tools-v2`, and `sakthai-context-7b-merged`), documented architecture classifications, weights packaging (Safetensors / GGUF / PEFT), empirical RL viability, and synchronized references in `README.md` and `CLAUDE.md`.
- **`sakthai-sft-training/tests/test_eval_cross_model.py` & `test_train_embedding.py`** — comprehensive unit test suite (18 tests) covering cross-model evaluation tool parsing (`<tool_call>` tags, JSON argument normalization, nested structures), argument and strict match scoring, percentage calculations, bilingual retrieval triplet integrity, and tool catalog schema validity.
- **`openenv-custom-training/tests/test_multi_env.py`** — 2 contract and unit tests for `multi_env.py`, validating type-isolated reward functions (`guess_reward`, `agent_tools_reward`), dataset schema generation (`prompt`, `environment`, `target`, `task`), and reward function list registration.
- **`openenv-custom-training/tests/test_agent_tools_wrapper.py`** — 6 contract and unit tests for `AgentToolEnv` in `agent_tools/wrapper.py`, covering 0-arg initialization, Google-style `Args:` docstrings, post-completion rejection on `run_command`/`write_file`/`read_file`, client step/result forwarding, `reward_func`, and `build_dataset`.
- **`openenv-custom-training/tests/test_train_select.py`** — 6 unit tests covering `train.py` dispatch logic (`_select` across `simple`, `agent_tools`, `mcp`, and invalid environment rejection), boolean environment variable parsing (`_bool_env`), and fallback observation rendering (`_obs_text`).
- **`openenv-multi-catalog-training/tests/test_multi_catalog_env.py` expansion** — 3 unit tests covering `run_servers.sh` CLI options (`--dry-run` and `--help`), step tools execution and state progression for `echo_say`, `sudoku_move`, `repl_execute`/`repl_finish`, and `sumo_set_phase`, and invalid reset routing validation, expanding the multi-catalog suite from 21 to 24 tests.
- **`sakthai-sft-training/tests/test_embedding_trainer.py`** — 4 unit tests covering `train-sakthai-embedding.py`, verifying tool triplet integrity, catalog alignment, bilingual evaluation query distribution, dataset building (`build_triplet_dataset`), and CLI argument parsing (`parse_args`).
- **`sakthai-sft-training/tests/test_ops_scripts.py` expansion** — added `test_smoke_test_check_argparse_with_env` to verify CLI `--help` preflight checks with environment variable injection.

### Fixed

- **`sakthai-sft-training/eval-cross-model.py` & `sakthai-agentic-eval-train/scripts/eval_bench_peft.py`** — added strict URL scheme verification (`https://` / `http://` only) across dataset downloads and HF router completions, remediating CWE-22 / Bandit B310 unexpected scheme risks.
- **`openenv-multi-catalog-training/a2a_agent/server.py`** — hardened default network binding from `0.0.0.0` to `127.0.0.1` with `A2A_HOST` / `A2A_PORT` environment variable support, remediating Bandit B104 unauthenticated public interface exposure.
- **`openenv-custom-training/tests/test_client.py`** — replaced hardcoded `/tmp/test.txt` path with isolated workspace paths, remediating Bandit B108 insecure temp directory risks.
- **`README.md`** — removed leftover merge conflict marker in benchmark section, updated test pass badges and tables to 148 passed tests across 5 workspace suites, and added Security Policy badge.
- **`SECURITY.md`** — updated git emergency response token search command to portable bash `grep -E`.
- **`sakthai-sft-training/eval-cross-model.py`** — decoupled top-level `urlopen` dataset downloading into lazy `load_test_rows()` function, eliminating import-time network I/O and enabling fully offline, deterministic unit testing on CPU runners.
- **`.github/workflows/lighteval.yml`** — removed redundant runner-side `pip install lighteval torch transformers accelerate datasets` (~1–2 GB download), aligning with `train.yml` and `eval.yml` to only install `huggingface_hub` on the GitHub Actions runner VM while remote execution continues on Hugging Face H200 GPUs.
- **`.github/workflows/benchmark-evaluation-manager-agent.yml`** — added `pull_request: branches: [main]` and `workflow_dispatch:` triggers to ensure the evaluation manager agent rubric benchmark and regression tests execute automatically on pull requests before merging.
- **`sakthai-sft-training/train-sakthai-embedding.py`** — registered missing `"translate_text"` in `TOOL_CATALOG_DOCS` to match training triplets.
- **`openenv-multi-catalog-training/run_servers.sh`** — granted executable permissions (`chmod +x`).
- **`.gitignore`** — added patterns for `.coverage`, `.coverage.*`, `htmlcov/`, `.pytest_cache/`, and `.ruff_cache/` to keep working tree clean during test and lint runs.
- **Dependency upgrade** — merged Dependabot update (#241) bumping `openenv==0.13.0`, `trl==1.18.1`, and `transformers>=5.23.0` across custom and multi-catalog training requirements, closing superseded PRs #240 and #239.

### Changed

- **`.github/workflows/verify-contracts.yml`** — added `openenv-custom-training/evaluate_baselines.py --episodes 5` automated gate and unified all 148 tests across 5 workspace suites.
- **Test suite expansion & Documentation synchronization** — increased passing test coverage to 148 tests across all five workspace suites, adding comprehensive "Security & Quality Assurance" architecture documentation in `README.md` and updating `CLAUDE.md`.

## 2026-11-16

### Added

- **`sakthai-agentic-eval-train/eval_sakthai_bench_colab.ipynb`** — free Colab T4 evaluation notebook for `Nanthasit/sakthai-bench-v2`, upgraded to use `eval_bench_peft.py` so both `-merged` models and bare PEFT `-tools` adapters can be scored cell-by-cell with a 20 GB VRAM guard for 7B.
- **`openenv-custom-training/env_mcp_task.py` & `tests/test_mcp_env.py`** — MCP-backed OpenEnv environment (`SakThaiMCPEnvironment(MCPEnvironment)`) and TRL `environment_factory` wrapper (`MCPTaskEnv`, `--env mcp` in `train.py`), plus CPU contract stubs for `openenv.core.env_server.mcp_types` and `MCPEnvironment` in `conftest.py`.
- **`openenv-custom-training/evaluate_baselines.py`** — CPU-runnable baseline policy competition (`AlwaysStay`/`AlwaysZero`, `Random`, `EpsilonGreedy`, `SmartHeuristic`) across `SimpleGuessEnv` and simulated 10×5 OpenSpiel Catch, reporting win rates, average steps, and GRPO group reward variance (`frac_reward_zero_std`).
- **`sakthai-sft-training/train-sakthai-embedding.py`** — PEP 723 script for fine-tuning and evaluating `Nanthasit/sakthai-embedding-multilingual` with `SentenceTransformerTrainer` and `MultipleNegativesRankingLoss` on bilingual (Thai + English) tool-retrieval triplets.
- **`openenv-multi-catalog-training/tests/test_a2a_agent.py`** — contract and lifecycle suite for `OpenEnvAgentExecutor`, testing `TaskUpdater` protocol methods (`submit`, `start_work`, `update_status`, `failed`, `cancel`), initial episode reset, `DataPart` action precedence, completion, and error rejection (closes `KNOWN_GAPS.yaml#a2a-sdk-method-names-unverified`).
- **`.claude/skills/` translation** — translated 5 high-value `.opencode/skills/` (`cycle-workflow`, `data-augmentation`, `training`, `eval`, `troubleshooting`) into native Claude Code skills with YAML frontmatter, execution workflows, and architectural guidance for SFT and OpenEnv RL pipelines.
- **`.github/workflows/bench-v3-publish.yml` & `sakthai-bench-v3` integration** — added scheduled weekly workflow submitting `eval_bench_peft.py` against `Nanthasit/sakthai-bench-v3` on HF Jobs, with dataset filename resolution (`data/sakthai-bench-v3.jsonl`) and split fallback in `eval_bench_peft.py` and `test_eval_bench_peft.py`.
- **`Nanthasit/sakthai-coder-browser` lightweight BrowserGym option** — added `BROWSERGYM_CODER_MODEL` and `--browsergym-model` choice (`default` / `coder`) in `openenv-custom-training/train.py`, enabling BrowserGym RL training to run on 1.5B Qwen2.5-Coder weights without the 20 GB VRAM requirement of 7B, guarded by `test_browsergym_contract.py`.
- **`Nanthasit/my-embedding-gemma` embedding backbone** — added `GEMMA_BASE_MODEL` constant and `--model-id` support in `sakthai-sft-training/train-sakthai-embedding.py` to enable fine-tuning and evaluating Gemma 3 dense embeddings alongside `sakthai-embedding-multilingual`.

### Fixed

- **`sakthai-agentic-eval-train/scripts/eval_bench_peft.py`** — added `_load_bench_dataset()` fallback to raw `data/test.jsonl` when `datasets.load_dataset` raises `ValueError: Feature type 'Json' not found` on Colab, with regression coverage in `test_eval_bench_peft.py`.
- **`openenv-multi-catalog-training/train_multi_env.py` & `a2a_agent/agent_executor.py`** — fixed `_format_openspiel_obs` truncation (`info_state[:32]` → `[:64]` with active non-zero indices and multi-player `current_player_id` / `opponent_last_action` fields) so the 50-cell Catch paddle row (indices 45–49) is no longer cut off; replaced `coding_env`'s `print(17 * 23)` placeholder task with a structured invoice-parsing function task (closes `KNOWN_GAPS.yaml#coding-env-placeholder`).
- **`sakthai-sft-training/eval-cross-model.py`** — added optional `INCLUDE_EXTERNAL_BASELINES=1` (`google/functiongemma-270m-it`) and `USE_HF_ROUTER=1` (`https://router.huggingface.co/v1`) remote inference fallback when HF Jobs returns `402 Payment Required`.
- **`sakthai-sft-training/train-sakthai-1.5b-v2.py:88`** — repointed missing `sakthai-combined-v8` to `Nanthasit/sakthai-combined-v10` (the latest existing Hub revision), preventing silent shard drops during training (closes `KNOWN_GAPS.yaml#dataset-v8-missing`).
- **`.ruff.toml` & style rule cleanup** — widened linter select to include `F401` (unused imports) and `F841` (unused variables) with exemptions preserving immutable `cycle-100-v*.py` snapshots and notebook cells byte-for-byte; resolved all 37 real findings across active files (closes `KNOWN_GAPS.yaml#ruff-style-rules`).

### Changed

- **Documentation audit and synchronization** — updated root `README.md`, `CLAUDE.md`, `openenv-custom-training/README.md`, `openenv-multi-catalog-training/README.md`, `openenv-multi-catalog-training/a2a_agent/README.md`, and `sakthai-agentic-eval-train/README.md` to reflect the 66 passed tests, `evaluate_baselines.py`, `eval_sakthai_bench_colab.ipynb` badges/links, MCP environment integration, and upgraded invoice-parsing task definitions.
- **Audit of stale publish helpers** — audited `sakthai-sft-training/push-v9-comprehensive.py` and `sakthai-sft-training/push-all-to-hub.py`, marking both explicitly as archived/deprecated with docstring explanations covering the superseded Hub lineage (`v10`) and uncommitted historical files.
- **Model catalog & dataset synchronization (`docs/MODELS.md`)** — updated Master Model Portfolio Matrix with `sakthai-coder-browser`, `sakthai-coder-browser-lora`, `my-embedding-gemma`, `sakthai-context-1.5b-merged-v2`, and `sakthai-context-0.5b-tools-sft-v2`; added new Core Datasets reference section (`sakthai-bench-v3`, `sakthai-combined-v12`, `sakthai-coder-browser`, `sakthai-github-actions-v2`).
- **Weekly Hub Monitor expansion (`.github/workflows/monitor.yml`)** — expanded weekly reporting step to enumerate downloads and likes across all 17 `Nanthasit` datasets in addition to models.
- **Configurable SFT augmentation (`sakthai-sft-training/train-sakthai-1.5b-v2.py`)** — added `AUGMENTED_DATASET` environment variable support allowing seamless upgrade from `v10` to `Nanthasit/sakthai-combined-v12`.

## 2026-11-13

### Changed

- **`README.md`** — refreshed live status table date to 2026-11-13, cleaned up duplicate status headings, and consolidated the dependency pass entry to record the latest updates (PRs #215, #216, #224).
- **`openenv-custom-training/requirements.txt` & `dependabot.yml`** — unified and deduplicated runtime requirements in `openenv-custom-training/requirements.txt`, updating `openenv==0.12.0`, `trl==1.17.1`, `transformers>=5.22.2`, and `datasets>=5.2.0`, removing historical duplicate lines caused by prior merge conflicts. Configured `groups.dependencies` for all `pip` ecosystems in `.github/dependabot.yml` to prevent duplicate/split PRs. Superseded and consolidated PRs #217-#223.
- **`codeql.yml` & `dependabot.yml`** — unified `github/codeql-action/init` and `github/codeql-action/analyze` to `33d7b85da1a04171e211befe2c739303b9a4c066` (`v4.39.3`), matching `upload-sarif` in `ossar.yml` and resolving the CodeQL version mismatch failure from split Dependabot PRs (#213, #214). Added `groups.github-actions` in `.github/dependabot.yml` so future GitHub Actions updates are grouped together in one PR.

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
