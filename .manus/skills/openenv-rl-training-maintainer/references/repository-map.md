# OpenEnv RL Training Repository Map

| Area | Location | Role |
|---|---|---|
| GRPO training | `openenv-custom-training/` | OpenEnv environment factories, training entrypoints, and reward extraction. |
| Agentic evaluation | `sakthai-agentic-eval-train/` | Evaluation scripts, findings, and model-selection evidence. |
| SFT | `sakthai-sft-training/` | Supervised fine-tuning and dataset preparation. |
| BrowserGym | `browsergym-space/` and related environment code | BrowserGym/MiniWoB++ environment assets and deployed-space contracts. |
| A2A | `a2a_agent/` | Agent-to-agent integration; SDK assumptions require live verification. |
| Contracts | `verify_grpo_contract.py`, `test_browsergym_contract.py`, `tests/` | CPU-safe checks intended to run without a GPU stack. |
| Automation | `.github/workflows/` | Contract checks, code scanning, dependency review, evaluations, training dispatch, and monitoring. |

## Local commands

Use the commands documented by the root README and affected subproject. The intended CPU-safe baseline is:

```bash
python3 verify_grpo_contract.py
uv run --with datasets --with pytest pytest test_browsergym_contract.py
```

The checkout may intentionally omit `torch`, `trl`, and other GPU dependencies. Do not install a large training stack or run a full training job merely to validate a contract change.

## Workflow classes

- Free PR/push checks: contract verification, CodeQL, dependency review, and related static checks.
- Scheduled or manual paid jobs: training, evaluation, lighteval, MCP-Bench, and other HF Jobs dispatches.
- Monitoring and maintenance: stale, summary, auto-update, and Hub-monitor workflows.

Read the workflow file itself for current trigger, secret, timeout, and artifact details; this map is not a substitute for live configuration.
