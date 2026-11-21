# CHANGELOG

Repo-level landed work, newest first. Empirical findings stay in
[`sakthai-agentic-eval-train/FINDINGS.md`](sakthai-agentic-eval-train/FINDINGS.md);
open items stay in [`PLAN.md`](PLAN.md) and [`docs/KNOWN_GAPS.yaml`](docs/KNOWN_GAPS.yaml).
When a gap closes, delete its PLAN.md / KNOWN_GAPS.yaml entry and add a line here.

---

## 2026-11-21

### Added

- **`openenv-custom-training/tests/test_multi_env.py`** — 2 contract and unit tests for `multi_env.py`, validating type-isolated reward functions (`guess_reward`, `agent_tools_reward`), dataset schema generation (`prompt`, `environment`, `target`, `task`), and reward function list registration.
- **`openenv-custom-training/tests/test_agent_tools_wrapper.py`** — 6 contract and unit tests for `AgentToolEnv` in `agent_tools/wrapper.py`, covering 0-arg initialization, Google-style `Args:` docstrings, post-completion rejection on `run_command`/`write_file`/`read_file`, client step/result forwarding, `reward_func`, and `build_dataset`.
- **`openenv-custom-training/tests/test_train_select.py`** — 6 unit tests covering `train.py` dispatch logic (`_select` across `simple`, `agent_tools`, `mcp`, and invalid environment rejection), boolean environment variable parsing (`_bool_env`), and fallback observation rendering (`_obs_text`).
- **`openenv-multi-catalog-training/tests/test_multi_catalog_env.py` expansion** — 3 unit tests covering `run_servers.sh` CLI options (`--dry-run` and `--help`), step tools execution and state progression for `echo_say`, `sudoku_move`, `repl_execute`/`repl_finish`, and `sumo_set_phase`, and invalid reset routing validation, expanding the multi-catalog suite from 21 to 24 tests.
- **`sakthai-sft-training/tests/test_embedding_trainer.py`** — 4 unit tests covering `train-sakthai-embedding.py`, verifying tool triplet integrity, catalog alignment, bilingual evaluation query distribution, dataset building (`build_triplet_dataset`), and CLI argument parsing (`parse_args`).
- **`sakthai-sft-training/tests/test_ops_scripts.py` expansion** — added `test_smoke_test_check_argparse_with_env` to verify CLI `--help` preflight checks with environment variable injection.

### Fixed

- **`sakthai-sft-training/train-sakthai-embedding.py`** — registered missing `"translate_text"` in `TOOL_CATALOG_DOCS` to match training triplets.
- **`openenv-multi-catalog-training/run_servers.sh`** — granted executable permissions (`chmod +x`).
- **`.gitignore`** — added patterns for `.coverage`, `.coverage.*`, `htmlcov/`, `.pytest_cache/`, and `.ruff_cache/` to keep working tree clean during test and lint runs.
- **Dependency upgrade** — merged Dependabot update (#241) bumping `openenv==0.13.0`, `trl==1.18.1`, and `transformers>=5.23.0` across custom and multi-catalog training requirements, closing superseded PRs #240 and #239.

### Changed

- **Test suite expansion** — increased passing test coverage from 108 to 130 tests across all five workspace suites, elevating code coverage to 70%.
- **Documentation synchronization** — updated `README.md` test badge and architecture table, and updated `CLAUDE.md` to reflect 130 passed tests.

## 2026-11-17

### Added

- **`sakthai-sft-training/tests/test_eval_scripts.py`** — 7 unit tests covering `contamination_audit.py` (`_content_str`, `_first_user_prompt`, `_normalize`, `_hash`, `get_training_specs`, and `generate_report`) and `dataset_composition_stats.py` (`_num_tool_calls`, `classify_row` across `simple`, `parallel`, `irrelevance`, `multi_turn`, and `other`).
- **`sakthai-sft-training/tests/test_ops_scripts.py`** — 5 unit tests covering `smoke_test.py` (`_check_python_parses`, `_check_pep723_header`), `submit_job.py` (`looks_like_debug` regex matching), and `weekly_ops_report.py` (`_fmt_duration`, `_bucket_script`).
- **`openenv-multi-catalog-training/tests/test_a2a_agent.py` client smoke test coverage** — 2 unit tests covering `client_smoke_test.py` (`send` and `main` round-trip), bringing the A2A suite to 11 tests.
- **`openenv-custom-training/tests/test_simple_guess_env.py`** — 8 contract and behavior unit tests for `SimpleGuessEnv` and `env_simple_task.py` covering 0-arg initialization, Google-style `Args:` docstrings, post-completion rejection, out-of-bounds parameter validation, attempt capping, bisection rollout, reward forwarding, and dataset construction.
- **`openenv-custom-training/tests/test_evaluate_baselines.py`** — 8 unit tests for `evaluate_baselines.py` validating `_compute_frac_zero_std` edge cases and group variances, `PolicyResult.win_rate` calculation, simulated Catch environment mechanics, and policy action selection (`Random`, `AlwaysStay`, `SmartCatch`, `EpsilonGreedy`, `AlwaysZero`, `OptimalBisect`).
- **`openenv-multi-catalog-training/tests/test_multi_catalog_env.py`** — 10 contract and lifecycle unit tests for `MultiCatalogEnv` in `train_multi_env.py`, validating tool schema docstrings (`Args:` blocks for `transformers.get_json_schema`), task routing isolation, post-completion call rejection, Catch paddle row preservation in `_format_openspiel_obs`, Atari RAM / SUMO formatting, per-task reward routing, and `build_dataset` generation.

### Fixed

- **`conftest.py` stub hardening** — added lightweight CPU stubs for `trl` (`GRPOConfig`, `GRPOTrainer`) and `a2a.client` (`A2ACardResolver`, `Client`), making all workspace test suites executable independently without ordering coupling on `test_browsergym_contract.py`.
- **`openenv-multi-catalog-training/a2a_agent/server.py`** — aligned coding skill description with the `CODING_EXPECTED` invoice parse-and-sum contract.
- **`openenv-multi-catalog-training/a2a_agent/client_smoke_test.py`** — added `trust_env=False` to `httpx.AsyncClient` to prevent broken environment proxy interference when connecting to localhost.
- **`sakthai-sft-training/scripts/ops/submit_job.py`** — added optional quote matching to `duckdb -c 'select hello'` debug pattern to accurately detect quoted duckdb hello-world commands and downgrade GPU flavors.
- **`openenv-multi-catalog-training/train_multi_env.py`** — upgraded `_format_openspiel_obs` from 32 to 64 floats with active non-zero index reporting and multi-player `current_player_id` / `opponent_last_action` metadata, preventing truncation of the Catch paddle row (indices 45–49); added safe attribute access in `_format_atari_obs` and `_format_sumo_obs`; added post-completion guards (`if self.done: raise ValueError(...)`) across all 9 tool methods; replaced shared list references in `build_dataset`.
- **`openenv-custom-training/env_simple_task.py`** — added post-completion guard (`if self.done: raise ValueError(...)`) to `guess()` to match repo contract and prevent altering attempts or state after an episode finishes.
- **`openenv-custom-training/agent_tools/wrapper.py`** — added post-completion guards to `run_command`, `write_file`, and `read_file` to reject actions immediately after episode termination; corrected return type annotation of `build_dataset`.
- **`openenv-custom-training/agent_tools/server/sandbox_env.py`** — added explicit `close()` cleanup method and previous workspace cleanup in `reset()` using `shutil.rmtree`, preventing temporary scratch directory leakage in long-running container sessions.
- **`sakthai-sft-training/push-all-to-hub.py`** — added explicit existence check for `training-data-ready.jsonl`, exiting cleanly with notice instead of raising an unhandled `FileNotFoundError`.
- **`verify_grpo_contract.py`** — simplified list comprehension in parameter extraction to `list(...)`.

### Changed

- **`openenv-multi-catalog-training/run_servers.sh`** — added `--dry-run` / `-n`, `--status` / `-s`, and `--help` / `-h` flags for safe inspection and container verification without requiring an active Docker daemon.
- **Repository documentation & CI synchronization** — updated `.github/workflows/verify-contracts.yml`, `README.md`, `CLAUDE.md`, `sakthai-sft-training/README.md`, `sakthai-agentic-eval-train/README.md`, and `.pre-commit-config.yaml` to reflect the 108 passed tests across 5 workspace suites, updated local test execution instructions, documented current ruff gate rules (`E9`, `F821`, `F811`, `F401`, `F841`), and clarified dataset Hub handles.

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
