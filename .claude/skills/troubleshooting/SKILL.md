---
name: troubleshooting
description: Comprehensive triage and diagnostic guide for errors in the SakThai / OpenEnv project. Use when diagnosing CUDA out-of-memory (OOM), missing secrets/tokens, HF Jobs failures, TRL API deprecations, Hugging Face Hub dataset loading issues, contract mismatches, or Kaggle execution timeouts.
---

# Troubleshooting Guide — SakThai & OpenEnv

## 1. CUDA & Memory Issues

### "CUDA out of memory" (OOM)
```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate ... MiB
```
**Remediation**:
- Reduce batch size: set `per_device_train_batch_size = 1`.
- Increase gradient accumulation steps to maintain effective batch size.
- Shorten context length: `max_seq_length = 1024` (or `max_completion_length = 64` for GRPO).
- Ensure `gradient_checkpointing = True` is set on the model / training config.
- Verify 4-bit quantization (NF4) is active via `BitsAndBytesConfig(load_in_4bit=True)`.
- In Colab/Kaggle, restart runtime and run `torch.cuda.empty_cache()` before loading large checkpoints.

### BitsAndBytes Not Supported
```text
bitsandbytes is not supported on Windows / No CUDA runtime found
```
**Remediation**:
- For local testing without GPU, use `train-sakthai-cpu.py` or local CPU test scripts.
- Run GPU workloads on Linux virtual environments, Kaggle T4, or HF Jobs.

---

## 2. Hugging Face & Authentication Issues

### "AssertionError: Set HF_TOKEN secret"
```text
AssertionError: Set HF_TOKEN secret: --secrets HF_TOKEN
```
**Remediation**:
- Local / CLI: Ensure `export HF_TOKEN="hf_..."` is in your environment.
- HF Jobs: Pass `--secrets HF_TOKEN` on the job submission CLI.
- GitHub Actions: Verify `secrets.HF_TOKEN` is configured in repository secrets.

### "402 Payment Required" on HF Jobs
```text
402 Client Error: Payment Required for url: https://huggingface.co/api/jobs
```
**Remediation**:
- See `docs/KNOWN_GAPS.yaml#hf-jobs-402`.
- `.github/scripts/hf-jobs-submit.sh` catches 402 and logs a warning with exit code 0 so CI pipelines do not fail.
- To execute real runs, add prepaid credit on Hugging Face or switch to free Kaggle T4 runtimes.

### Missing Dataset on Hub (e.g. `sakthai-combined-v8`)
```text
HfHubHTTPError: 404 Client Error: Entry Not Found for url: ...
```
**Remediation**:
- Repoint dataset loading to the verified available revision on the Hub (`Nanthasit/sakthai-combined-v10`).

### Feature type 'Json' not found in `load_dataset`
```text
ValueError: Feature type 'Json' not found
```
**Remediation**:
- In `eval_bench_peft.py` and similar scripts, fall back to reading raw `data/test.jsonl` lines directly via `json.loads` if arrow schema resolution fails on older `datasets` releases.

---

## 3. TRL & Model Training Errors

### `cannot import name 'SFTTrainer'` / API Mismatches
**Remediation**:
- TRL underwent major refactoring between 0.12 and 1.x.
- Use `SFTConfig` instead of passing `TrainingArguments` directly to `SFTTrainer`.
- Pass tokenizer via `processing_class=tokenizer`, not `tokenizer=tokenizer`.
- Pin compatible releases in `requirements.txt`: `trl>=1.17.0`, `transformers>=5.22.0`.

### `<tool_call>` Tokens Not Generated
**Causes & Remediation**:
- Greedy decoding vs sampling: Set `do_sample=False` or `temperature=0.1` during evaluation.
- Tokenizer chat template: Verify `apply_chat_template(messages, tools=tools)` is formatted with valid tool definitions.
- Missing special tokens: Ensure tool special tokens (`<tool_call>`, `</tool_call>`) are registered in tokenizer vocabulary.

---

## 4. Contract & Testing Issues

### OpenSpiel Catch Paddle Observation Truncation
- Catch observation states require a 64-element array (10×5 grid + paddle row + metadata).
- If indexing only `[:32]`, paddle position at indices 45–49 is discarded, preventing RL agents from observing paddle location. Always slice `[:64]`.

### A2A Protocol Stubs & Tests
- Ensure `TaskUpdater` protocol lifecycle calls (`submit`, `start_work`, `update_status`, `complete`) handle asynchronous message delivery.
- Echo environments complete in 1 step (`episode.done = True`), whereas multi-turn environments like Sudoku require step-based done conditions.
