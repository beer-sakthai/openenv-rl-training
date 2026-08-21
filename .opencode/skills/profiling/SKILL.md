---
name: profiling
description: Use when the user mentions performance, speed, latency, throughput, tokens per second, benchmarking inference, optimizing generation, or comparing model speed across SakThai sizes. Phase: PROFILE.
---

# Performance Profiling — SakThai Inference

## Expected throughput

| Model | Format | Hardware | Tokens/sec | Latency (first token) |
|---|---|---|---|---|
| 0.5B | GGUF Q4_K_M | CPU (4 cores) | ~15 tok/s | ~200 ms |
| 0.5B | GGUF Q4_K_M | CPU (8 cores) | ~25 tok/s | ~120 ms |
| 0.5B | BF16 | T4 GPU | ~50 tok/s | ~60 ms |
| 1.5B | GGUF Q4_K_M | CPU (4 cores) | ~6 tok/s | ~500 ms |
| 1.5B | GGUF Q4_K_M | CPU (8 cores) | ~12 tok/s | ~250 ms |
| 1.5B | BF16 | T4 GPU | ~30 tok/s | ~100 ms |
| 7B | GGUF Q4_K_M | CPU (8 cores) | ~3 tok/s | ~1.5 s |
| 7B | BF16 | T4 GPU | ~15 tok/s | ~200 ms |
| 7B | BF16 | A10G | ~40 tok/s | ~80 ms |

## Profiling script
```python
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Nanthasit/sakthai-context-1.5b-merged"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

prompt = "What's the weather in Bangkok? " * 50  # ~100 tokens
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Warmup
_ = model.generate(**inputs, max_new_tokens=10)

# Benchmark
num_tokens = 100
start = time.time()
outputs = model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False)
elapsed = time.time() - start
generated = outputs[0][inputs["input_ids"].shape[1]:].shape[0]

print(f"Generated {generated} tokens in {elapsed:.2f}s = {generated/elapsed:.1f} tok/s")
```

## Optimization techniques

### 1. Quantization
| Method | Speedup | Quality loss |
|---|---|---|
| No quantization (BF16) | 1x | None |
| 8-bit (bitsandbytes) | ~1.3x | Negligible |
| 4-bit NF4 | ~1.5x | Minimal |
| GGUF Q4_K_M | ~2x (CPU) | Small |
| GGUF Q8_0 | ~1.5x (CPU) | Very small |

### 2. Batch inference
For eval, batch multiple prompts:
```python
inputs = tokenizer(prompts, return_tensors="pt", padding=True)
outputs = model.generate(**inputs, max_new_tokens=200)
```
Batch size 16 is optimal for T4; higher values may OOM.

### 3. vLLM for production
```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model Nanthasit/sakthai-context-1.5b-merged \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9
```
vLLM provides 2-5x throughput over raw transformers via PagedAttention.

### 4. Flash Attention 2
```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
```
Requires CUDA and `pip install flash-attn`. ~1.3x speedup on long sequences.

### 5. Speculative decoding
Draft model (0.5B) → target model (7B):
```
# TGI supports this natively with --speculative-model
```

## Memory profiling
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Cached:    {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

## Bottleneck diagnosis
| Symptom | Likely cause | Fix |
|---|---|---|
| High TTFT | Model loading | Quantize, use GGUF |
| Low tok/s | Memory bandwidth | Reduce precision |
| High variance | CPU throttling | Check thermals |
| OOM mid-generation | KV cache | Reduce max_new_tokens |
