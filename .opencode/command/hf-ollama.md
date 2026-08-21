---
description: Set up SakThai models with Ollama for local inference — pull existing GGUF, create Modelfile, test tool-calling.
agent: general
---

Help the user run SakThai models locally with Ollama. Read the ollama skill for reference.

1. **Check what's available**:
   - 0.5B: Q4_K_M GGUF exists (~398 MB)
   - 1.5B: Q4_K_M GGUF exists (~986 MB) — most popular
   - 7B: **No GGUF yet** — offer to create it

2. **Pull existing model**:
   ```bash
   ollama pull hf.co/Nanthasit/sakthai-context-1.5b-merged
   ```

3. **Or create from local GGUF** with a Modelfile:
   ```dockerfile
   FROM ./sakthai-1.5b-Q4_K_M.gguf
   PARAMETER temperature 0.0
   TEMPLATE """<|im_start|>system\n{{ .System }}<|im_end|>\n<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n"""
   SYSTEM "You are SakThai, a tool-calling assistant."
   ```
   ```bash
   ollama create sakthai-1.5b -f Modelfile
   ```

4. **Test tool-calling**:
   ```bash
   curl http://localhost:11434/api/chat -d '{
     "model": "sakthai-1.5b",
     "messages": [{"role": "user", "content": "Weather in Bangkok?"}],
     "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"city": {"type": "string"}}}}]
   }'
   ```

5. Optionally guide through creating the missing 7B GGUF (most impactful addition).

Return step-by-step instructions tailored to the user's model choice.
