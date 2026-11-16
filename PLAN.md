# PLAN — SakThai end-to-end training repo

**Status:** consolidation of `beer-sakthai/SakThai-Training` into this repo landed
2026-08-21. This file is the open-work record. Resolved work moves to
[`CHANGELOG.md`](CHANGELOG.md); empirical decisions live in
[`sakthai-agentic-eval-train/FINDINGS.md`](sakthai-agentic-eval-train/FINDINGS.md); the
parseable gap index is [`docs/KNOWN_GAPS.yaml`](docs/KNOWN_GAPS.yaml).

When a bullet lands, delete it here (not strike it) and add a line to CHANGELOG.md.

---

## Open — hygiene / lint

- [ ] Repoint `augmented-output/push-augmented.py` at the Hub so the committed
      `.jsonl` payloads under `augmented-output/` and `safety-quality-fixes/` can be
      removed. Blocked on a push to the Hub first; see `KNOWN_GAPS.yaml#committed-dataset-payloads`.

## Open — infra / dependencies

- [ ] **Add prepaid HF Jobs credit.** Sole blocker on the four HF-Jobs workflows:
      `402 ... Pre-paid credit balance is insufficient`, confirmed in the 2026-09-14
      `lighteval` job log after a successful login. See `KNOWN_GAPS.yaml#hf-jobs-402`.
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
- [ ] Dispatch a single `--env browsergym` episode against `openenv/browsergym_env`
      and compare the observation + reward against `_BrowserGymTaskEnv`'s inferred
      assumptions. See `KNOWN_GAPS.yaml#browsergym-env-wrapper-unverified`.


---

### Deferred / not in scope

- Adding `pyproject.toml` — CLAUDE.md is explicit this is not a package.
- Unifying routing column names (`environment` / `env` / `task`) — deliberate divergence
  across the three workspaces; each one is self-consistent.
- Renaming `sakthai_grpo_colab.ipynb` or `sakthai-agentic-eval-train/` — breaks
  Colab/Kaggle badges that point at `main`.
