# Consolidation & Evaluation Analysis — SakThai-Training → openenv-rl-training

**Date:** 2026-08-21 · **Status:** executed — Phases 1, 2, 4 done; Phase 3 GitHub deletion
awaits `delete_repo` scope (`gh auth refresh -h github.com -s delete_repo`), local archive done.
**Repos:** `beer-sakthai/openenv-rl-training` (survivor) · `beer-sakthai/SakThai-Training` (deleted on GitHub, archived at `/home/beern/archive/SakThai-Training`)
**Owner:** beer-sakthai (HF: [Nanthasit](https://huggingface.co/Nanthasit))

This document is the "from start to finish" answer sheet for the consolidation: **what** we
are doing, **how**, **how long**, **what it costs**, **suggestions/recommendations**, and
which **models / subagents / HF tools & skills** can be wired into this repo.

---

## 1. What are we doing?

One repo must survive: **`openenv-rl-training`**. It already hosts both halves of the
SakThai training pipeline after the 2026-08-21 consolidation PR:

| Half | Workspace | Contents |
|---|---|---|
| SFT | `sakthai-sft-training/` | QLoRA (0.5B/1.5B/7B), 10-cycle augmentation, bench harness, evaluators, ops tooling, Colab notebook — **was `SakThai-Training`** |
| RL | `openenv-custom-training/` | Tier A/B environments + BrowserGym MiniWoB++ |
| RL | `openenv-multi-catalog-training/` | 8-catalog multi-env GRPO runner + A2A server |
| RL | `sakthai-agentic-eval-train/` | The pipeline that actually ran + `FINDINGS.md` |

`SakThai-Training` was the SFT-only repo. Almost everything unique was already migrated.
The remaining delta is small and now enumerated:

| Artifact | Status |
|---|---|
| Training/eval scripts, bench harness, data, ops tooling | ✅ already in `sakthai-sft-training/` |
| `AGENTS.md` | ✅ folded into `CLAUDE.md` |
| `SECURITY.md`, `docs/HF_HUB_IMPROVEMENTS.md` | ✅ at repo root |
| `submit_job.py` (secret redaction) | ✅ newer version in repo (better: redacts env/secrets) |
| `README.md` | ✅ superseded by the rewritten two-half README |
| **`.eval_results/` (3 benchmark YAMLs)** | ⚠️ **only real gap** — tracked in old repo, gitignored here → preserved into `sakthai-sft-training/sakthai-cycle-bench/eval_results/` |
| Repo itself | ❌ **delete** (GitHub + local) — user-directed, irreversible |

**Deliverables this effort produces:**
1. Plan files on `main` (this PR): `PLAN.md` + `docs/CONSOLIDATION.md`.
2. A verified, evaluated, fully consolidated `openenv-rl-training`.
3. `SakThai-Training` gone from GitHub and disk.
4. A documented evaluation report (evidence gates) + improvement list.

---

## 2. HOW? — step-by-step (the execution plan)

**Phase 1 — Plan PR (this change).** Rewrite `PLAN.md`; add this analysis doc; open PR to
`openenv-rl-training`; merge to `main` (user instruction: plan lands before execution).

**Phase 2 — Finish consolidation.**
1. Copy `SakThai-Training/.opencode/skills/huggingface/fixes/.eval_results/*.yaml` →
   `sakthai-sft-training/sakthai-cycle-bench/eval_results/` (rename avoids root
   `.gitignore`'s `.eval_results/` rule).
2. Diff sweep: `git ls-files` of old repo vs new repo — every tracked file must be present
   or intentionally superseded (documented above).
3. Fix any references to the old repo URL in docs (except the historical note).

**Phase 3 — Delete `SakThai-Training`.**
1. Final parity confirmation (automated diff, shown to user).
2. `gh repo delete beer-sakthai/SakThai-Training --yes` (irreversible — done only after
   user confirmation at that step).
3. Remove local `/home/beern/SakThai-Training` (history still lives in GitHub's recycle
   window for 90 days and in the local clone if kept).

**Phase 4 — Evaluate (evidence gates, all local / free).**
- `python3 verify_grpo_contract.py` (passes; TRL section skips without `trl`).
- `uv run --with datasets --with pytest pytest test_browsergym_contract.py` (3 passed).
- `pytest` for `sakthai-agentic-eval-train/tests/` + `openenv-custom-training/tests/`.
- YAML-parse every `.github/workflows/*.yml`.
- Grep for dangling `SakThai-Training` references.

**Phase 5 — Improve (driven by Phase 4 findings).**
- Add `HF_TOKEN` + `STEP_SECURITY_API_KEY` as repo secrets (unblocks 5 HF-Jobs workflows).
- Report findings; fix surfaced issues; update PLAN.md checkboxes.
- Future: weekly `bench-v3` regen workflow, Claude Code skills translation, 7B GRPO run.

---

## 3. How long does it take?

| Phase | Time | Who |
|---|---|---|
| Plan + analysis docs | ~30–45 min | agent (this PR) |
| Finish consolidation (parity, eval_results, refs) | ~15 min | agent |
| Delete GitHub repo + local | ~5 min | agent + 1 user confirm |
| Local evaluation (contract tests, pytest, workflow lint) | ~20–30 min | agent |
| Improvements + docs + final report | ~30–60 min | agent |
| **Total hands-on** | **~2–3 hours** | — |
| Optional GPU work (7B GRPO, bench regen) | hours–days, paid | HF Jobs / Colab |

---

## 4. Cost

| Item | Cost |
|---|---|
| This consolidation + evaluation (CPU-only) | **$0** |
| GitHub repo ops (delete, PR, Actions) | **$0** (public repo, free tier) |
| `verify-contracts.yml` CI | **$0** (GitHub Actions CPU) |
| HF Jobs GPU (eval 0.5B/1.5B: `l4x1`) | ~$0.5–1.5/hr |
| HF Jobs GPU (7B bench eval, 7B GRPO: `a100-large` 80GB) | ~$3–8/hr · a real GRPO run: **$10–60** (hundreds of steps) |
| HF Jobs subscription | currently blocked: `402 Payment Required` — no spend possible until payment method added |
| Kaggle T4 (free weekly quota) | **$0** — the practical free path for SFT + small GRPO |
| Colab free tier | **$0** for short runs |

**Bottom line:** this whole consolidation costs **nothing**. The only money is future GPU
training, and FINDINGS.md says the free path (Kaggle/Colab) is viable for the 7B GRPO run.

---

## 5. Suggestions & recommendations

**Recommended (do these):**
1. **Merge plan PR to `main` first** — plan-before-code (this PR does exactly that).
2. **Delete `SakThai-Training` only after Phase 2 parity diff** — we have an explicit
   checklist, so deletion is provably safe.
3. **Add `HF_TOKEN` repo secret now** — it unblocks 5 of 6 workflows at zero cost; without
   it the workflows are dead weight.
4. **Keep `train.yml` on `workflow_dispatch`-only** — prevents surprise GPU bills.
5. **Preserve `.eval_results/`** into `sakthai-cycle-bench/eval_results/` — they are the
   only tracked benchmark record of `sakthai-bench-v2` (48.2% selection on 1.5B).

**Suggested (when budget/time allows):**
6. Weekly `bench-v3` regeneration workflow (mirrors `monitor.yml` cadence).
7. Real 7B GRPO via the consolidated Colab notebook (free T4 path) → push
   `Nanthasit/sakthai-context-7b-tools-grpo`.
8. Claude Code skills (`.claude/skills/`) translated from `.opencode/skills/` so SakSee /
   SakJules / other CLIs can drive the pipeline.
9. Replace `coding_env` placeholder task; verify `a2a_agent` against a live `a2a-sdk`.

**Not recommended:**
- Unifying the incompatible TRL pinsets (SFT 0.19.1 vs RL current) in one requirements
  file — breaks both halves (documented in CLAUDE.md).
- Re-adding GitHub's suggested pylint/super-linter templates — removed deliberately.
- "Refactoring" `cycle-100-v2..v10.py` into one file — the snapshots ARE the record.

---

## 6. Use models to run the ecosystem? Subagents?

**Yes — and it's already partly set up.** The repo's `.opencode/` ships 25 commands + 35
skills precisely so a model agent can drive the pipeline.

| Question | Answer |
|---|---|
| Can a model run this ecosystem? | **Yes.** Local CPU part: `verify_grpo_contract.py` + tests. Heavy part: `hf jobs uv run …` (needs payment) or Kaggle/Colab notebooks. |
| Which model to use for the training targets? | **`Nanthasit/sakthai-context-7b-tools`** — the only viable GRPO target (FINDINGS.md). 0.5B is SFT-only. |
| Which model to use AS the agent driving the repo? | Any tool-calling model the agent CLI runs on (the repo is model-agnostic); for orchestrating HF Jobs, an agent with `hf` CLI + `HF_TOKEN` is enough. |
| Subagents? | **Recommended for the heavy/parallel work:** (1) one subagent per evaluation lane (contract tests, pytest suites, workflow lint, docs grep) — they run in parallel and report evidence; (2) a "hub publish" subagent for dataset/model card work; (3) a "bench runner" subagent for GPU jobs. This session uses parallel evaluation lanes. |
| Model-as-a-service for evals? | lighteval / inspect-ai / lm-eval-harness via HF Jobs (paid) or local CPU for tiny models. The `lighteval.yml` workflow already exists. |

---

## 7. Hugging Face profile (Nanthasit) — tools & skills to add to this repo

Live inventory taken 2026-08-21 (via HF API): **19 models, 19 datasets, 1 private Space**
(`sakthai-sft-trainer`), plus the deployed `Nanthasit/browsergym-env` Space referenced by
`browsergym-space/`. The repo already knows most of it; candidates to wire in:

### Already referenced by this repo
- Models: `sakthai-context-{0.5b,1.5b,7b}-tools` / `-merged` (+`-v2`), `sakthai-coder-*`, `sakthai-vision-7b`, `sakthai-tts-model`, `sakthai-embedding-multilingual`.
- Datasets: `sakthai-combined-v{6,7,10,12}`, `sakthai-bench-v{1,2,3}`, `eval_results`, `sakthai-openenv-training` (version pin), `hermes-tool-use-rl-env`.
- Space: `Nanthasit/browsergym-env` (BrowserGym server, live).

### Tools/skills worth adding to this repo (pick per agent CLI)
| Skill / tool | Why it fits | Where it lives |
|---|---|---|
| `huggingface-llm-trainer` (TRL SFT/GRPO/DPO + GGUF) | Matches `train-*.py` + `create-7b-gguf.py` | `~/.agents/skills/huggingface-llm-trainer` |
| `huggingface-community-evals` (inspect-ai/lighteval) | Matches `lighteval.yml` + `eval-*.py` | `~/.agents/skills/huggingface-community-evals` |
| `huggingface-datasets` (Dataset Viewer API) | Matches the augmentation/audit scripts | `~/.agents/skills/huggingface-datasets` |
| `huggingface-local-models` + `hf-mem` | GGUF quantization + memory planning for 7B | `~/.agents/skills/*` |
| `huggingface-hub` / `hub-api` skills | Publish + card fixes (`push-*.py`) | already mirrored in `.opencode/skills/hub-api` |
| `huggingface-gradio` / `huggingface-spaces` | Build/run the `browsergym-env` + trainer Spaces | `~/.agents/skills/huggingface-*` |
| `hf jobs uv run` patterns | All HF-Jobs workflows | documented in CLAUDE.md § Workflows |
| MCP: this session's `hf-mcp-server` | Search/inspect/attach models, datasets, spaces, run jobs — used to verify the inventory above | opencode config |

**Recommendation:** do not bloat the repo with more vendored skills (it already has 35 in
`.opencode/skills/`). Instead add a `docs/HF_TOOLING.md` index pointing at the global
skill library above per agent CLI, and translate the 5 core `.opencode/skills/` into
`.claude/skills/` (existing follow-up).

---

## 8. Exit criteria (definition of done)

1. `PLAN.md` + `docs/CONSOLIDATION.md` merged to `main`.
2. Diff sweep shows zero unique tracked files lost (all preserved or superseded).
3. `.eval_results/` preserved under `sakthai-sft-training/sakthai-cycle-bench/eval_results/`.
4. All local evaluation gates pass (contract checks, pytest, workflow YAML parse, no
   dangling references).
5. `beer-sakthai/SakThai-Training` deleted on GitHub; local checkout removed.
6. PLAN.md checkboxes updated; README/CLAUDE.md wording matches reality.
7. Final report delivered (this doc + chat summary).
