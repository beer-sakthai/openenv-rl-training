# PLAN — SakThai end-to-end training repo

**Status:** consolidation of `beer-sakthai/SakThai-Training` into this repo landed 2026-08-21.
Supersedes the pre-consolidation four-task GRPO improvement list — everything on that list
is either done, obsolete, or moved into the "Landed" table below.

Empirical decisions live in [`sakthai-agentic-eval-train/FINDINGS.md`](sakthai-agentic-eval-train/FINDINGS.md);
this file is the intent + progress record.

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

---

## Follow-ups (separate PRs, not blocked on consolidation)

### FINDINGS-driven

- [ ] **Real 7B GRPO run.** The proof-of-signal was 40 steps (`grad_norm` 0.25–0.49, non-zero reward variance). A real run needs hundreds of steps on `Nanthasit/sakthai-context-7b-tools` (the only viable GRPO target in this family per FINDINGS). Use `sakthai-agentic-eval-train/sakthai_grpo_colab.ipynb` as the entry point; push to `Nanthasit/sakthai-context-7b-tools-grpo`.
- [ ] **Retire `sakthai-context-0.5b-tools` as a GRPO target.** FINDINGS shows zero reward variance → zero gradient. Mark it as "SFT-only" in `sakthai-sft-training/README.md`; leave the SFT recipe alone.
- [ ] **Bench-v3 nightly regen.** `.github/workflows/monitor.yml` runs weekly Hub health-check; extend it (or add a sibling `bench-v3-publish.yml`) to regenerate `Nanthasit/sakthai-bench-v3` on the same cadence. Depends on `HF_TOKEN`.

### Repo hygiene

- [ ] Add `HF_TOKEN` (and `STEP_SECURITY_API_KEY` referenced by `eval.yml`) as GitHub repo secrets. Until then only `verify-contracts.yml` runs.
- [ ] Archive `beer-sakthai/SakThai-Training` on GitHub (Settings → Archive) once its cleanup PR merges. User-driven.
- [ ] Translate a subset of `.opencode/skills/` into Claude Code skills (`.claude/skills/*/SKILL.md`) — start with `cycle-workflow`, `data-augmentation`, `training`, `eval`, `troubleshooting`. Separate PR; do not merge with routine changes.
- [ ] Verify a catalog Docker image tag in `openenv-multi-catalog-training/run_servers.sh` live and lock it (currently unverified per `CLAUDE.md` § Known open items).
- [ ] Replace `coding_env`'s placeholder task (`print(17 * 23)`) with a real coding-tool-use task; carry `a2a_agent/coding_env` along.
- [ ] Execute `a2a_agent/` against a live `a2a-sdk` install and confirm the `TaskUpdater` method names; update its README once verified.

### Deferred / not in scope

- Adding `pyproject.toml` — CLAUDE.md is explicit this is not a package.
- Unifying routing column names (`environment` / `env` / `task`) across workspaces — deliberate divergence, do not touch.
- Renaming `sakthai_grpo_colab.ipynb` or `sakthai-agentic-eval-train/` — breaks Colab/Kaggle badges.
