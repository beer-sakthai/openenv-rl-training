# Training and Release Safety

## Job classification

Before launching anything, classify the action:

| Class | Typical location | Cost and verification |
|---|---|---|
| Contract check | `verify_grpo_contract.py`, browser contract tests | CPU-safe and suitable for local/PR verification. |
| Proof-of-signal run | Training/eval scripts with a small step budget | GPU-dependent; proves gradients or wiring, not model quality. |
| Full training | HF Jobs workflows or `hf jobs uv run` | Paid/remote; requires explicit user intent, secrets, timeout, and destination review. |
| Evaluation | Agentic eval, lighteval, MCP-Bench workflows | Remote or GPU-dependent; record baseline, model revision, environment, and score artifacts. |

## HF Jobs checklist

1. Confirm the requested environment, GPU flavor, timeout, base model, dataset, step/episode limits, and push destination.
2. Confirm the required `HF_TOKEN` secret and account/job availability. A `402 Payment Required` response means the job did not run successfully; do not report it as a code failure or model result.
3. Review environment variables and generated artifacts before dispatch. Avoid placing tokens or private data in command lines, logs, or committed files.
4. After completion, record the job URL/id, model revision, dataset revision, metrics, artifact paths, and whether the run was complete or interrupted.
5. Update the relevant findings or model card only with evidence from the actual run.

## Model publishing

Treat `TRAIN_PUSH_TO`, Hub repositories, dataset names, and model revisions as release destinations. Verify the destination before a push and do not overwrite a shared model without explicit authorization. Keep training source, configuration, and reproducibility metadata alongside the release record.
