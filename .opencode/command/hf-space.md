---
description: Build interactive Gradio Spaces for SakThai — tool-calling demo, vision demo, TTS demo, or leaderboard. All existing spaces are static HTML.
agent: general
---

Help create interactive HF Spaces for the SakThai family. Current spaces (all static HTML) need upgrading:
- `sakthai-tts` → Gradio TTS demo (Kokoro-82M, CPU-friendly)
- `sakthai-leaderboard` → Gradio live benchmark viewer
- `sakthai-vision-demo` → Gradio vision demo (LLaVA GGUF, needs T4 GPU)
- New: `sakthai-demo` → Tool-calling chat with model selector

1. Read the gradio skill for reference code.
2. For the **tool-calling demo** (flagship), scaffold:
   - `app.py` with model selector dropdown (0.5B CPU / 1.5B GPU)
   - Tool JSON input box
   - Multi-turn conversation support
   - `requirements.txt` with torch, transformers, gradio, accelerate
3. For **vision demo**, use llama.cpp Python bindings with T4 GPU.
4. For **TTS demo**, use Kokoro on CPU (free tier).
5. Push via HF Hub UI or `HfApi.upload_folder` with `repo_type="space"`.

Return complete file contents and deployment steps for each space type.
