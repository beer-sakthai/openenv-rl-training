# PLAN — SakThai end-to-end training repo

**Status:** consolidation of `beer-sakthai/SakThai-Training` into this repo landed 2026-08-21.
This file is the intent + progress record; empirical decisions live in
[`sakthai-agentic-eval-train/FINDINGS.md`](sakthai-agentic-eval-train/FINDINGS.md).

Active effort (2026-08-21, evening): **consolidation completion** — finish folding every
remaining artifact of the retired repo into this one, delete `SakThai-Training` entirely
(GitHub + local), evaluate the merged repo end-to-end, and close the hygiene follow-ups
that a real evaluation can land today. See [`docs/CONSOLIDATION.md`](docs/CONSOLIDATION.md)
for the full analysis (what / how / time / cost / suggestions).

---

## Landed 2026-08-21 (consolidation PR)

- [x] `sakthai-sft-training/` created; SFT scripts, cycle-100 augmentation, evaluators, bench harness, gap-fill data, ops tooling, and SFT Colab notebook moved in.
- [x] `.github/workflows/` seeded from SakThai-Training (`train`, `eval`, `lighteval`, `mcp-bench`, `monitor`); paths rewritten to `sakthai-sft-training/…`.
- [x] `.github/workflows/verify-contracts.yml` added — runs `verify_grpo_contract.py` and `test_browsergym_contract.py` on every PR.
- [x] `train.yml` set to `workflow_dispatch`-only to avoid unintentional paid GPU spend.
- [x] `.opencode/` (25 commands + 35 skills) lifted verbatim; no path rewrites needed.
- [x] `SECURITY.md` and `docs/HF_HUB_IMPROVEMENTS.md` lifted; `AGENTS.md` folded into `CLAUDE.md` § "The SFT half".
- [x] `README.md` and `CLAUDE.md` rewritten to cover both halves + document the end-to-end pipeline.
- [x] **Bug fix:** `openenv-multi-catalog-training/train_multi_env.py:main()` `vllm_server_url` → `vllm_server_host` + `vllm_server_port`.
- [x] **Bug fix:** `openenv-custom-training/train.py` `--env browsergym` — added `browsergym_env` dep to PEP 723 header + `requirements.txt`.
- [x] Upstream sync: 185 commits pulled (`main` → `96580a1`), including root `requirements.txt` and unit tests under `sakthai-agentic-eval-train/tests/` and `openenv-custom-training/tests/`.
- [x] `.eval_results/` benchmark YAMLs preserved from SakThai-Training → moved to `sakthai-sft-training/sakthai-cycle-bench/eval_results/` (they were tracked in the old repo but gitignored by this repo's root `.gitignore`).

## Consolidation-completion plan (this PR + follow-ups)

### Phase 1 — Plan + PR (this PR, `claude/consolidate-final-*`)

- [x] Rewrite `PLAN.md` as the single plan record.
- [x] Add `docs/CONSOLIDATION.md` — full analysis: What / HOW / timeline / cost / suggestions / recommendations / models / subagents / HF tools & skills.
- [x] Open PR to `beer-sakthai/openenv-rl-training`, merge to `main` (per user instruction, plan lands before execution).

### Phase 2 — Finish consolidation (execution) — ✅ landed via PR #54

- [x] Preserve `sakthai-sft-training/sakthai-cycle-bench/eval_results/` (3 benchmark YAMLs from the old repo, byte-identical).
- [x] Verify no tracked file of `beer-sakthai/SakThai-Training@master` is missing here (diff sweep: 0 gaps; `AGENTS.md` intentionally folded into `CLAUDE.md`).
- [x] Update `CLAUDE.md` + `README.md` references from "archive" to "deleted" once the old repo is gone.

### Phase 3 — Delete `beer-sakthai/SakThai-Training`

- [x] Confirm content parity one final time (all unique tracked files preserved or superseded — diff sweep 0 gaps).
- [~] `gh repo delete beer-sakthai/SakThai-Training --yes` (irreversible — user-directed; **blocked on `delete_repo` scope**: run `gh auth refresh -h github.com -s delete_repo` interactively, then the delete command).
- [x] Local `/home/beern/SakThai-Training` archived → `/home/beern/archive/SakThai-Training` (full git history kept on disk).
- [x] Update this PLAN.md "Landed" table + README/CLAUDE.md wording to reflect deletion.

### Phase 4 — Evaluate everything (evidence gates) — ✅ all passed 2026-08-21

- [x] `python3 verify_grpo_contract.py` — passes, TRL section skips when absent.
- [x] `uv run --with datasets --with pytest pytest test_browsergym_contract.py` — 4 passed.
- [x] `pytest` under `sakthai-agentic-eval-train/tests/` and `openenv-custom-training/tests/` — 19 passed (after making them CPU-runnable: mocked `torch`/`transformers`/`huggingface_hub`/`hermes-tool-use-rl-env` imports).
- [x] Workflow lint: all 14 `.github/workflows/*.yml` parse clean.
- [x] Docs consistency: no dangling references to `SakThai-Training` except historical notes.

### Phase 5 — Improvements (drive from evaluation results)

- [ ] Add `HF_TOKEN` (and `STEP_SECURITY_API_KEY` referenced by `eval.yml`) as repo secrets — unblocks the 5 HF-Jobs workflows. **User action:** `gh secret set HF_TOKEN --repo beer-sakthai/openenv-rl-training` (and `STEP_SECURITY_API_KEY` if eval.yml is used).
- [ ] Fix anything evaluation surfaces (report findings in PR/comments, not silent edits).
- [ ] Extend `monitor.yml` or add `bench-v3-publish.yml` for weekly `sakthai-bench-v3` regeneration (depends on `HF_TOKEN`).
- [ ] Translate a subset of `.opencode/skills/` into Claude Code skills (`.claude/skills/*/SKILL.md`) — `cycle-workflow`, `data-augmentation`, `training`, `eval`, `troubleshooting` first. Separate PR.

---

## Follow-ups (not blocked on consolidation; from FINDINGS.md + hygiene list)

- [ ] **Real 7B GRPO run.** Proof-of-signal was 40 steps. A real run needs hundreds of steps on `Nanthasit/sakthai-context-7b-tools` (the only viable GRPO target). Use `sakthai-agentic-eval-train/sakthai_grpo_colab.ipynb`; push to `Nanthasit/sakthai-context-7b-tools-grpo`.
- [ ] **Retire `sakthai-context-0.5b-tools` as a GRPO target.** Zero reward variance → zero gradient. Mark "SFT-only" in `sakthai-sft-training/README.md`.
- [ ] Verify a catalog Docker image tag in `openenv-multi-catalog-training/run_servers.sh` live and lock it.
- [ ] Replace `coding_env`'s placeholder task (`print(17 * 23)`) with a real coding-tool-use task.
- [ ] Execute `a2a_agent/` against a live `a2a-sdk` install and confirm `TaskUpdater` method names.
- [ ] HF Jobs currently returns `402 Payment Required` — five HF-Jobs workflows fail until payment + `HF_TOKEN` land.

### Deferred / not in scope

- Adding `pyproject.toml` — CLAUDE.md is explicit this is not a package.
- Unifying routing column names (`environment` / `env` / `task`) — deliberate divergence.
- Renaming `sakthai_grpo_colab.ipynb` or `sakthai-agentic-eval-train/` — breaks Colab/Kaggle badges.
