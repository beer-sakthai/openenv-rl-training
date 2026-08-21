---
description: Benchmark inference speed (tokens/sec, latency) for any SakThai model across different hardware and quantization formats.
agent: general
---

Help the user profile their SakThai model's inference performance.

1. Read the profiling skill for reference throughput tables.
2. Determine what they want to measure:
   - Which model? (0.5B, 1.5B, 7B)
   - Which format? (BF16, GGUF Q4_K_M, 8-bit)
   - Which hardware? (CPU cores, GPU type)
   - What metric? (tokens/sec, first-token latency, memory)

3. Generate a profiling script adapted to their setup:
   ```python
   import torch, time
   from transformers import AutoModelForCausalLM, AutoTokenizer
   model_id = "..."
   tokenizer = AutoTokenizer.from_pretrained(model_id)
   model = AutoModelForCausalLM.from_pretrained(model_id, ...)
   # ... benchmark loop
   ```

4. Suggest optimizations if results are below expected:
   - Quantization (GGUF, bitsandbytes)
   - Flash Attention 2
   - vLLM for production
   - Batch inference for eval

Return the profiling script and expected vs actual results table.
