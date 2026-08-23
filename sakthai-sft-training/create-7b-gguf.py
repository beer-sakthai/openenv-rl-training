#!/usr/bin/env python3
"""
Generate GGUF for sakthai-context-7b-merged.
This is a SCRIPT (instructions + runner) - needs llama.cpp installed.

The 1.5B GGUF has 1,599 downloads - the most downloaded artifact.
The 7B has 0 GGUF downloads because none exists. This closes that gap.

Steps:
1. Install llama.cpp
2. Download the model
3. Convert to GGUF
4. Quantize (Q4_K_M, Q5_K_M)
5. Upload to Hugging Face

Usage:
  uv run python create-7b-gguf.py          # Show instructions
  # or run with actual commands on Linux/WSL
"""
import os, subprocess, platform

MODEL = "Nanthasit/sakthai-context-7b-merged"
GGUF_REPO = "Nanthasit/sakthai-context-7b-merged"  # upload to same repo

print("=" * 60)
print("7B GGUF Conversion Guide")
print("=" * 60)
print(f"""
Model: {MODEL}
Size: 7.62B params, ~15 GB BF16 -> ~4 GB Q4_K_M

Why: The 1.5B GGUF is #1 download (1,599). 7B GGUF will be #2.

=== Step 1: Install llama.cpp ===
  git clone https://github.com/ggerganov/llama.cpp
  cd llama.cpp
  cmake -B build
  cmake --build build --config Release

=== Step 2: Download model weights ===
  pip install huggingface_hub
  huggingface-cli download {MODEL} --local-dir ./sakthai-7b

=== Step 3: Convert to GGUF FP16 ===
  python convert_hf_to_gguf.py ./sakthai-7b \\
    --outfile ./sakthai-7b-fp16.gguf

=== Step 4: Quantize (recommended levels) ===
  # Best quality/size balance:
  ./build/bin/quantize ./sakthai-7b-fp16.gguf \\
    ./sakthai-7b-q4_k_m.gguf q4_K_M
  
  # Higher quality (larger):
  ./build/bin/quantize ./sakthai-7b-fp16.gguf \\
    ./sakthai-7b-q5_k_m.gguf q5_K_M

=== Step 5: Upload to Hugging Face ===
  huggingface-cli upload {GGUF_REPO} ./sakthai-7b-q4_k_m.gguf \\
    --repo-type model

=== Expected sizes ===
  FP16: ~15 GB
  Q4_K_M: ~4.4 GB
  Q5_K_M: ~5.4 GB

=== Expected performance ===
  CPU (8 cores) Q4: ~3 tok/s
  GPU (T4) Q4: ~15 tok/s
  GPU (A10G) Q4: ~35 tok/s
""")
