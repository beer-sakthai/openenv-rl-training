---
name: openenv-rl-training-maintainer
description: "Maintain, test, and safely integrate changes in beer-sakthai/openenv-rl-training. Use for OpenEnv environments, GRPO/SFT training, BrowserGym tasks, agentic evaluation, Hugging Face Jobs workflows, contract tests, model publishing, CI, and release or branch operations in this repository."
---

# OpenEnv RL Training Maintainer

Maintain this repository as a training and evaluation workspace rather than a general application. Keep CPU contract checks deterministic and lightweight, keep GPU/Hugging Face Jobs execution explicit, and never represent an unrun training job as a verified result.

## Start every task

1. Work from the repository root. Inspect the current branch, worktree, remotes, recent commits, and relevant `CLAUDE.md`/`README.md` guidance.
2. Classify the change as an environment contract, training script, evaluation harness, dataset/SFT pipeline, model publishing workflow, CI workflow, or documentation.
3. Read the local README for the affected directory before editing. Preserve the distinction between locally runnable contract checks and GPU-only jobs.
4. Inspect workflow triggers and required secrets before changing `.github/workflows/`.
5. Plan multi-step work and make the smallest change that satisfies the request.

## Repository surfaces

- `openenv-custom-training/`: OpenEnv-compatible GRPO training, environment factories, and reward extraction.
- `sakthai-agentic-eval-train/`: agentic evaluation and training utilities; consult its findings before selecting a model or benchmark.
- `sakthai-sft-training/`: supervised fine-tuning and dataset preparation.
- `browsergym-space/` and related environment directories: BrowserGym/MiniWoB++ tasks and deployed-space contracts.
- `a2a_agent/`: agent-to-agent integration; treat SDK assumptions as unverified until executed.
- `test_*_contract.py`, `verify_grpo_contract.py`, and `tests/`: CPU-safe contract and regression checks.
- `.github/workflows/`: free verification workflows plus paid HF Jobs dispatch workflows.

Use [repository-map.md](references/repository-map.md) for current commands, environment boundaries, and workflow classification. Use [training-and-release.md](references/training-and-release.md) for job launch, model publishing, and safety rules.

## Implementation rules

- Keep contract tests free of GPU, `torch`, `trl`, and `datasets` requirements where the existing design intends CPU-only verification.
- Keep environment factories and reward extraction deterministic, observable, and covered by focused contract tests.
- Treat training hyperparameters, base models, datasets, reward functions, and push destinations as explicit configuration. Do not silently change defaults.
- Do not add tokens, credentials, or paid-job assumptions to source. Use repository secrets and documented environment variables.
- Separate generated artifacts, caches, checkpoints, and model outputs from source changes. Review `.gitignore` and release manifests before committing artifacts.
- Update the nearest README or findings document when behavior, benchmark interpretation, model selection, or operational status changes.

## Verification workflow

1. Run `git diff --check` and focused CPU contract tests first.
2. Use the repository’s documented local commands, commonly:

```bash
python3 verify_grpo_contract.py
uv run --with datasets --with pytest pytest test_browsergym_contract.py
```

3. Do not attempt full training locally when the checkout intentionally lacks GPU dependencies. Instead validate configuration, imports, contract behavior, and workflow YAML.
4. For workflow changes, inspect permissions, triggers, secret names, job cost, timeout, artifact paths, and the target branch. Use GitHub Actions as the final verification.
5. Record whether a result is a contract-test result, a short proof-of-signal run, a full training run, or an evaluation baseline. Do not conflate them.

## GitHub integration

- Treat `main` as protected. Use a feature branch, commit with a clear message, push it, open a PR, and wait for required checks.
- Do not bypass required checks or manually claim green CI. Inspect PR status with `gh pr view` before merging.
- For HF Jobs workflows, verify that the user has explicitly requested a paid dispatch and that required `HF_TOKEN`/billing conditions are available. A `402 Payment Required` result is an account/job limitation, not a code-pass result.
- Delete temporary branches only after their commits are merged and verified as ancestors of `main`.

## Completion report

Report changed files, local contract results, remote CI results, whether any GPU or paid job was actually run, model/dataset destinations, PR and merge commit, and the final branch state. Keep known open items visible rather than hiding them in a green status summary.
