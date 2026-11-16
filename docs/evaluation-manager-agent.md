# Evaluation Manager Agent Guide

This guide explains how to configure and extend the repository-specific Claude Code agent at `.claude/agents/openenv-rl-training-evaluation-manager.md`.

The agent manages evaluation work in `beer-sakthai/openenv-rl-training`. It is intentionally focused on evidence, reproducibility, and safe execution boundaries rather than on running arbitrary training jobs.

## What the agent is for

Use the agent for work involving:

- Single-shot `sakthai-bench-v2` evaluation and PEFT adapter loading.
- Agentic, multi-turn tool-use evaluation against `hermes-tool-use-rl-env`.
- Comparisons across models, adapters, checkpoints, datasets, renderers, or SFT/GRPO stages.
- Lighteval, MCP-Bench, ZeroGPU, Colab, Kaggle, and Hugging Face Jobs evaluation workflows.
- Evaluation provenance, transcript diagnosis, result interpretation, and updates to durable findings.

It should not be used to claim benchmark results from an unrun job, silently change evaluation methodology, or dispatch paid infrastructure without explicit authorization.

## Agent configuration

The agent uses YAML frontmatter followed by its system prompt:

```yaml
---
name: openenv-rl-training-evaluation-manager
description: "Use this agent when managing evaluation pipelines ..."
model: inherit
color: cyan
tools: ["Read", "Write", "Edit", "Grep", "Glob", "Bash"]
---
```

### Frontmatter fields

- **`name`** is the stable identifier. Keep it lowercase, use hyphens, and do not rename it casually because invocation references may depend on it.
- **`description`** controls discovery and triggering. Start with `Use this agent when`, mention the repository and evaluation scope, and include concrete `<example>...</example>` phrases for common requests.
- **`model`** is `inherit` by default so the agent follows the parent session’s model. Change this only when there is a documented capability or cost reason.
- **`color`** is a UI label. `cyan` identifies this as an analysis and evaluation agent.
- **`tools`** follows least privilege. The current set supports repository inspection, edits, validation, and Git operations without granting unrelated capabilities.

The body is the agent’s system prompt. Keep it below the validator’s 10,000-character limit and organize it into discoverable sections.

## Prompt structure

The current prompt has seven important sections:

1. **When to invoke** lists representative single-shot, agentic, comparison, and remote-operation scenarios.
2. **Core responsibilities** defines provenance, evaluator selection, reproducibility, and durable-record rules.
3. **Repository evaluation map** names the canonical scripts, findings, README, and remote workflows.
4. **Execution and safety rules** separates CPU checks from GPU or paid work and prevents secret leakage.
5. **Analysis process** gives the agent a repeatable inspect, validate, execute, compare, and record sequence.
6. **Pull request workflow** requires a feature branch, checks, mergeability, and explicit merge confirmation.
7. **Output format** requires a report with provenance, metrics, artifacts, cost boundaries, limitations, and PR status.

When adding guidance, place it in the section where another maintainer would look for it. Avoid duplicating the entire repository README or historical benchmark tables in the prompt; link or name the durable source instead.

## Extending the agent for a new evaluator

When a new evaluator or benchmark is added:

1. Add the evaluator path and its purpose to **Repository evaluation map**.
2. Document the required model, adapter, dataset, task scope, renderer, environment revision, and output variables.
3. State whether it is CPU-safe, GPU-only, remote, paid, scheduled, or workflow-dispatched.
4. Add the evaluator to the **When to invoke** scenarios if users are likely to request it directly.
5. Add the expected artifact and metric format to the **Output format** guidance.
6. Add or update focused tests for the evaluator itself before changing the agent prompt.
7. Update `sakthai-agentic-eval-train/README.md` or the nearest durable documentation when the evaluator is a supported repository surface.

For example, a new benchmark entry should identify the command or script, its required environment variables, whether it supports PEFT adapters, its scoring contract, and where results are stored. Do not add a bare name with no execution or provenance guidance.

## Extending safety and provenance rules

Every evaluation result should be reproducible from a provenance tuple containing at least:

- Model ID or local checkpoint and adapter/base relationship.
- Dataset and revision, task subset, and split.
- Renderer or chat-template mode.
- Environment revision and evaluator commit.
- Generation, sampling, batching, and timeout settings.
- Hardware, dependency versions, and job ID when remote.
- Result, transcript, and upload destinations.

If a new workflow can spend money, access a secret, publish to the Hub, or overwrite an artifact, add an explicit review step to the prompt. Keep the rule operational: specify what must be confirmed and what failure state must be reported. For example, a `402 Payment Required` response means that no benchmark run occurred.

Never add tokens, private URLs, credentials, or copied benchmark outputs to the agent file. Historical results belong in `FINDINGS.md` or a result artifact with its provenance.

## Testing changes to the agent

The regression tests are in `sakthai-agentic-eval-train/tests/test_evaluation_manager_agent.py`. They verify:

- Frontmatter identity and trigger examples.
- Coverage of the repository’s canonical evaluation surfaces.
- Provenance, renderer, payment, secret, and reporting safeguards.
- Required prompt sections.

Run the focused tests:

```bash
uv run --with pytest pytest \
  sakthai-agentic-eval-train/tests/test_evaluation_manager_agent.py -q
```

Run the repository’s full CPU gate before submitting a change:

```bash
uv run --with ruff ruff check .
python3 verify_grpo_contract.py
uv run --with datasets --with pytest --with pydantic --with httpx \
  pytest test_browsergym_contract.py \
         sakthai-agentic-eval-train/tests/ \
         openenv-custom-training/tests/ -q
```

Validate the agent structure directly as well:

```bash
bash /home/ubuntu/skills/agent-development/scripts/validate-agent.sh \
  .claude/agents/openenv-rl-training-evaluation-manager.md
```

These checks do not run a model benchmark. GPU, HF Jobs, and paid evaluation remain separate operations with their own authorization and provenance requirements.

## Benchmarking agent coverage

The deterministic benchmark at `sakthai-agentic-eval-train/scripts/benchmark_evaluation_manager_agent.py` scores the agent prompt against four repository scenarios: single-shot evaluation, agentic evaluation, pipeline comparison, and remote evaluation operations. It also checks execution boundaries, provenance requirements, and delivery workflow guidance. It does not call an LLM, run a model, access Hugging Face, or measure conversational quality; it is a prompt-coverage regression benchmark.

Run the human-readable report with:

```bash
python3 sakthai-agentic-eval-train/scripts/benchmark_evaluation_manager_agent.py
```

The command exits nonzero if a scenario is missing required guidance or the overall coverage score falls below 90%. Use `--json` for automation or CI artifact collection:

```bash
python3 sakthai-agentic-eval-train/scripts/benchmark_evaluation_manager_agent.py --json
```

When extending the agent, add a scenario term only when the wording represents a real repository contract. Update the benchmark tests when adding a new category, and avoid turning historical metric values into prompt requirements.

## Continuous integration

`.github/workflows/benchmark-evaluation-manager-agent.yml` runs on every push. It uses a pinned checkout, Python 3.11, and the repository’s hardened runner convention, then runs the JSON benchmark followed by the two focused regression-test modules. The workflow is CPU-only, requires no Hugging Face token, and does not start model evaluation or paid infrastructure.

Keep this workflow path-filter-free so changes to the agent, benchmark, tests, documentation, or repository configuration all exercise the same coverage gate. If the benchmark contract changes, update the script and tests together and confirm the workflow still reports a nonzero exit code for missing required guidance.

## Pull request checklist

Before opening a PR for an agent or guide change:

- Keep the change scoped to the agent, its tests, and relevant documentation.
- Review the diff for credentials, generated artifacts, stale metrics, and accidental evaluator changes.
- State exactly which local checks ran and whether any GPU or paid job ran.
- Use a feature branch targeting `main`.
- Wait for all applicable CI checks and resolve failures on the same branch.
- Do not merge an agent change until the exact merge action has been explicitly confirmed.
