---
name: troubleshooting
description: Use when the user encounters errors, bugs, failures in training, eval, HF Jobs, Kaggle, model loading, tokenizer issues, CUDA/CPU problems, or any unexpected behavior in the SakThai pipeline. Phase: TROUBLESHOOT.
---

# Troubleshooting — SakThai Common Issues

## Training issues

### "CUDA out of memory"
```
RuntimeError: CUDA out of memory. Tried to allocate ... MiB
```
**Causes**: Batch size too large, sequence length too long, gradient checkpointing off.
**Fixes**:
- Reduce `per_device_train_batch_size` (2 → 1)
- Reduce `max_seq_length` (2048 → 1024)
- Ensure `gradient_checkpointing=True`
- Use 4-bit quantization (already set in QLoRA)
- Switch to smaller model (1.5B → 0.5B)

### "HF_TOKEN not set"
```
AssertionError: Set HF_TOKEN secret: --secrets HF_TOKEN
```
**Fix**: Add `--secrets HF_TOKEN` to your HF Jobs command:
```bash
hf jobs uv run --flavor a10g-small --timeout 6h --secrets HF_TOKEN train-sakthai-1.5b-v2.py
```

### Training hangs on "Loading checkpoint shards"
**Cause**: First-time download of base model.
**Fix**: Wait (can take 5-10 min on slow connections). For subsequent runs, cache:
```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./cache/qwen-1.5b
```

### "transformers import error: cannot import name 'SFTTrainer'"
**Cause**: Wrong TRL version.
**Fix**: Ensure TRL >= 0.12, < 0.20:
```bash
pip install "trl>=0.19,<0.20"
```

### BitsAndBytes not found
```
bitsandbytes is not supported on Windows
```
**Fix**: Use CPU training (0.5B only) or run on Linux (HF Jobs, Kaggle).

## Eval issues

### "torch.cuda.is_available() returns False"
**Fix**: You're on CPU. Eval runs very slowly. Use HF Jobs or Kaggle.

### Model outputs garbage text
**Cause**: Wrong tokenizer (using base model tokenizer instead of fine-tuned).
**Fix**: Always use `tokenizer.from_pretrained(MODEL_ID)` where MODEL_ID is the fine-tuned repo, not the base.

### "<tool_call>" not appearing in output
**Cause**: Model defaulting to chat mode, temperature too high.
**Fixes**:
- Set `do_sample=False` (greedy decoding)
- Ensure tools are passed in the chat template
- Check training data has enough tool-calling examples

## HF Jobs issues

### "Job failed unexpectedly"
**Fix**: Check job logs:
```bash
hf jobs list
hf jobs logs <job-id>
```
Common causes: timeout (increase `--timeout`), OOM (smaller flavor), missing secret.

### "No space left on device"
**Fix**: The HF Jobs temp directory is limited. Use Storage Buckets for large datasets:
```bash
hf buckets sync ./local-dir hf://buckets/sakthai-training-bucket/
```

## Kaggle issues

### "GPU quota exceeded"
**Fix**: You have 30 hrs/week. Switch to HF Jobs or reduce epochs.

### "Internet disconnected"
Kaggle notebooks lose internet after ~1 hour of inactivity. Add a keep-alive cell:
```python
import time; [time.sleep(60) for _ in range(120)]  # 2 hour keep-alive
```

## Model card issues

- **Download badges showing wrong numbers**: Use dynamic shields: `https://img.shields.io/badge/dynamic/json?url=https%3A//huggingface.co/api/models/Nanthasit/sakthai-context-1.5b-merged&query=%24.downloads&label=downloads`
- **Widget not appearing**: Ensure model has an inference provider or `pipeline_tag` is set
- **model-index not showing**: Use new `.eval_results/` format instead of old YAML
