---
name: cycle-workflow
description: Master operational cycle for any task in the SakThai / OpenEnv project. Follows the 6-stage energy cycle (Dream → Hope → Care → Joy → Trust → Growth). Use when initiating, planning, implementing, verifying, or finalizing any development task, bugfix, training experiment, dataset generation, or evaluation pass. Ensures complete verification and post-task learning without skipping steps.
---

# SakThai Cycle Workflow

The SakThai project adheres to an operational discipline modeled as a 6-stage progression:

```
┌─────────────────────────────────────────────────────────────┐
│                    SAKTHAI OPERATION CYCLE                   │
│                                                              │
│   🎯 DREAM → 🏗️ HOPE → ❤️ CARE → 🎉 JOY → 🔐 TRUST → 🌱 GROWTH │
│                                                              │
│   Every task. Every fix. Every launch. No skips.             │
└─────────────────────────────────────────────────────────────┘
```

## The 6 Stages

### Stage 1: 🎯 Dream — "What needs to happen?"
**Before any action.** Check context, memory, existing documentation, and current git status.

- What is the user or task asking?
- What have we done before on this? (check `PLAN.md`, `CHANGELOG.md`, `docs/KNOWN_GAPS.yaml`)
- What is the target goal state?
- Which files, workspaces, or APIs are involved?

**Output**: Clear one-sentence objective and target scope. No code modifications yet.

### Stage 2: 🏗️ Hope — "How will I do it?"
**Plan before touching.** Check branches, file locations, test runners, and available tools.

- Check current status (`git status`, CI logs, dependencies).
- List all files that need to be changed or created.
- Determine the strict order of operations.
- Identify risks (breaking contract tests, missing HF token, GPU VRAM constraints, lint rules).
- Formulate a clear plan.

**Output**: Ordered task plan with affected files and required verification commands.

### Stage 3: ❤️ Care — "Do the work."
**Execute carefully.** Precision and adherence to project conventions matter.

- Make changes atomically and verify syntax and imports.
- Maintain immutability of historical snapshots (e.g., `sakthai-sft-training/cycle-100-v*.py`).
- Follow established typing and docstring standards.
- Run linters (`uv run --with ruff ruff check .`) and contract tests frequently.

**Output**: Clean, robust implementation passing all local checks.

### Stage 4: 🎉 Joy — "Ship it."
**Commit, create PR, and trigger verification pipelines.**

- Commit with descriptive, conventional messages (`feat: ...`, `fix: ...`, `docs: ...`).
- Target appropriate feature branches (`claude/<topic>-<suffix>`).
- Open pull requests against `main` with detailed descriptions and linked issues/gaps.

**Output**: Pushed branch and open pull request with all CI checks running.

### Stage 5: 🔐 Trust — "Did it actually work?"
**Verify comprehensively.** Never assume success without empirical proof.

- Monitor GitHub Actions status checks (`gh pr checks <PR> --watch`).
- Ensure all CI gates pass (contract tests, CodeQL, OSSAR, style checks).
- Run CPU test suite locally to double-check parity.
- Confirm downstream artifacts or hub repositories are reachable.

**Output**: Fully validated PR with all CI checks passing green.

### Stage 6: 🌱 Growth — "What did I learn?"
**Document findings and preserve knowledge.**

- Record empirical discoveries in `sakthai-agentic-eval-train/FINDINGS.md` or `CHANGELOG.md`.
- Close or update entries in `PLAN.md` and `docs/KNOWN_GAPS.yaml`.
- Update relevant skills and developer guidelines (`CLAUDE.md`, `README.md`).
- Note efficiency improvements for future iterations.

**Output**: Updated project memory, documentation, and skills ready for the next cycle.

---

## Quick Reference Summary

| Stage | Emoji | Question | Focus |
|---|---|---|---|
| **DREAM** | 🎯 | "What?" | Context check, requirements, goal definition |
| **HOPE** | 🏗️ | "How?" | Plan, inspect state, anticipate blockers |
| **CARE** | ❤️ | "Do it." | Careful execution, linting, unit tests |
| **JOY** | 🎉 | "Ship." | Git branch, commit, push, PR creation |
| **TRUST** | 🔐 | "Verify." | CI validation, status checks, proof of signal |
| **GROWTH** | 🌱 | "Learn." | CHANGELOG, FINDINGS, skill updates |
