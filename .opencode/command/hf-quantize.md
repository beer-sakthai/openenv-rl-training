---
description: Create GGUF quantized versions of any SakThai model for Ollama, llama.cpp, and LM Studio. Most needed for 7B.
agent: general
---

Help create GGUF quants for SakThai models. Current state:
- 0.5B merged: **has GGUF** (Q4_K_M) ✅
- 1.5B merged: **has GGUF** (Q4_K_M) ✅ (most downloaded at 1,599)
- 7B merged: **no GGUF** ❌ — most impactful to add

1. Read the inference skill for reference.
2. For the **7B model** (priority target):
   ```bash
   # Install llama.cpp
   git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
   cmake -B build && cmake --build build --config Release

   # Download
   huggingface-cli download Nanthasit/sakthai-context-7b-merged --local-dir ./sakthai-7b

   # Convert to GGUF FP16
   python convert_hf_to_gguf.py ./sakthai-7b --outfile ./sakthai-7b-fp16.gguf

   # Quantize (recommended: q4_K_M and q5_K_M)
   ./build/bin/quantize ./sakthai-7b-fp16.gguf ./sakthai-7b-q4_k_m.gguf q4_K_M
   ./build/bin/quantize ./sakthai-7b-fp16.gguf ./sakthai-7b-q5_k_m.gguf q5_K_M

   # Upload to the merged model's repo
   huggingface-cli upload Nanthasit/sakthai-context-7b-merged ./sakthai-7b-q4_k_m.gguf .
   ```
3. For Ollama Modelfile:
   ```dockerfile
   FROM ./sakthai-7b-q4_k_m.gguf
   TEMPLATE "{{ .Prompt }}"
   ```
   ```bash
   ollama create sakthai-7b -f Modelfile
   ```

Return the complete step-by-step guide. For 0.5B or 1.5B, adapt the model name accordingly.
