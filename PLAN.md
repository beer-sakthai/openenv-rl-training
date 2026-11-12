# PLAN — SakThai end-to-end training repo

**Status:** consolidation of `beer-sakthai/SakThai-Training` into this repo landed
2026-08-21. This file is the open-work record. Resolved work moves to
[`CHANGELOG.md`](CHANGELOG.md); empirical decisions live in
[`sakthai-agentic-eval-train/FINDINGS.md`](sakthai-agentic-eval-train/FINDINGS.md); the
parseable gap index is [`docs/KNOWN_GAPS.yaml`](docs/KNOWN_GAPS.yaml).

When a bullet lands, delete it here (not strike it) and add a line to CHANGELOG.md.

---

## Open — hygiene / lint

- [ ] Widen the ruff select list to the style rules (F541/F401/F841, ~115 findings).
      Safe subset path: exclude `cycle-100-v*.py` snapshots, enable F401 + F841 only
      (skip F541 — f-string noise is low signal), fix the remaining ~30 real findings.
      The pinned snapshots must stay byte-identical.
- [ ] Repoint `augmented-output/push-augmented.py` at the Hub so the committed
      `.jsonl` payloads under `augmented-output/` and `safety-quality-fixes/` can be
      removed. Blocked on a push to the Hub first; see `KNOWN_GAPS.yaml#committed-dataset-payloads`.
- [ ] Translate a subset of `.opencode/skills/` into Claude Code skills
      (`.claude/skills/*/SKILL.md`) — `cycle-workflow`, `data-augmentation`, `training`,
      `eval`, `troubleshooting` first.
- [ ] Audit stale publish helpers. `push-v9-comprehensive.py` was never run;
      `push-all-to-hub.py` was broken until 2026-09-16. If v9/v11 won't be pushed,
      mark the scripts "deprecated — see KNOWN_GAPS.yaml" or delete them.

## Open — infra / dependencies

- [ ] **Add prepaid HF Jobs credit.** Sole blocker on the four HF-Jobs workflows:
      `402 ... Pre-paid credit balance is insufficient`, confirmed in the 2026-09-14
      `lighteval` job log after a successful login. See `KNOWN_GAPS.yaml#hf-jobs-402`.
- [ ] Push `sakthai-combined-v8` (or repoint `train-sakthai-1.5b-v2.py:88` at `v10`).
      v8/v9/v11 do not exist on the Hub; `train.yml` silently drops the v8 shard via
      a `try/except`. See `KNOWN_GAPS.yaml#dataset-v8-missing`.
- [ ] Redeploy `browsergym-space/` or keep the upstream default. `Nanthasit/browsergym-env`
      returns 404; `train.py --env browsergym` now points at `openenv/browsergym_env`.
      See `KNOWN_GAPS.yaml#browsergym-space-not-deployed`.
- [ ] Verify a catalog Docker image tag in `openenv-multi-catalog-training/run_servers.sh`
      live and lock it. See `KNOWN_GAPS.yaml#catalog-docker-tags-unverified`.

## Open — real runs

- [ ] **Real 7B GRPO run.** Proof-of-signal was 40 steps. A real run needs hundreds of
      steps on `Nanthasit/sakthai-context-7b-tools` (the only viable GRPO target in
      this family — settled in FINDINGS.md). Use `sakthai-agentic-eval-train/sakthai_grpo_colab.ipynb`;
      push to `Nanthasit/sakthai-context-7b-tools-grpo`. See `KNOWN_GAPS.yaml#grpo-proof-run-too-short`.
- [ ] Replace `coding_env`'s placeholder task (`print(17 * 23)`) with a real
      coding-tool-use task. See `KNOWN_GAPS.yaml#coding-env-placeholder`.
- [ ] Execute `a2a_agent/` against a live `a2a-sdk` install and confirm `TaskUpdater`
      method names. See `KNOWN_GAPS.yaml#a2a-sdk-method-names-unverified`.
- [ ] Dispatch a single `--env browsergym` episode against `openenv/browsergym_env`
      and compare the observation + reward against `_BrowserGymTaskEnv`'s inferred
      assumptions. See `KNOWN_GAPS.yaml#browsergym-env-wrapper-unverified`.

## Open — monitoring

- [ ] Extend `monitor.yml` or add `bench-v3-publish.yml` for weekly `sakthai-bench-v3`
      regeneration (depends on `HF_TOKEN`).

---

### Deferred / not in scope

- Adding `pyproject.toml` — CLAUDE.md is explicit this is not a package.
- Unifying routing column names (`environment` / `env` / `task`) — deliberate divergence
  across the three workspaces; each one is self-consistent.
- Renaming `sakthai_grpo_colab.ipynb` or `sakthai-agentic-eval-train/` — breaks
  Colab/Kaggle badges that point at `main`.
