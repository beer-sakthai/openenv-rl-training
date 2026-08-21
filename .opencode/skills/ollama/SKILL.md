---
name: ollama
description: Use when the user mentions Ollama, llama.cpp, local inference, on-device, edge deployment, Modelfile, or running SakThai models locally on CPU/GPU. Phase: DEPLOY.
---

# Ollama — Local SakThai Deployment

## Models available for Ollama

| Model | GGUF | Size | RAM needed | Status |
|---|---|---|---|---|
| `sakthai-context-0.5b-merged` | Q4_K_M | ~398 MB | ~1 GB | ✅ Published |
| `sakthai-context-1.5b-merged` | Q4_K_M | ~986 MB | ~2 GB | ✅ Published |
| `sakthai-context-7b-merged` | **Missing** | ~4 GB (est) | ~6 GB | ❌ Needs creation |

## Pull existing models
```bash
ollama pull hf.co/Nanthasit/sakthai-context-1.5b-merged
```

## Create custom Modelfile
```dockerfile
FROM ./sakthai-context-1.5b-Q4_K_M.gguf

PARAMETER temperature 0.0
PARAMETER top_p 0.9

TEMPLATE """{{- range .Messages }}
{{- if eq .Role "system" }}<|im_start|>system
{{ .Content }}<|im_end|>
{{- else if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{- else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}<|im_end|>
{{- end }}
{{- end }}
<|im_start|>assistant
"""

SYSTEM """You are SakThai, a tool-calling AI assistant. When a user asks a task that requires a function call, respond with:
<tool_call>{"name": "<function_name>", "arguments": {<args>}}</tool_call>
If no tool is needed, respond normally."""
```

```bash
ollama create sakthai-1.5b -f Modelfile
ollama run sakthai-1.5b
```

## Tool-calling with Ollama API
```bash
curl http://localhost:11434/api/chat -d '{
  "model": "sakthai-1.5b",
  "messages": [{"role": "user", "content": "What is the weather in Bangkok?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"city": {"type": "string"}}}}]
}'
```

## Ollama + smolagents
```python
from smolagents import OllamaModel, CodeAgent, tool

model = OllamaModel(model_id="sakthai-1.5b", num_predict=256)

@tool
def get_weather(city: str) -> str:
    """Get weather."""
    return f"Weather in {city}: sunny"

agent = CodeAgent(tools=[get_weather], model=model)
agent.run("Weather in Bangkok")
```

## Creating GGUF for Ollama (7B priority)
```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release

# Download model
huggingface-cli download Nanthasit/sakthai-context-7b-merged --local-dir ./sakthai-7b

# Convert + quantize
python convert_hf_to_gguf.py ./sakthai-7b --outfile sakthai-7b-fp16.gguf
./build/bin/quantize sakthai-7b-fp16.gguf sakthai-7b-q4_k_m.gguf q4_K_M

# Upload to existing model repo
huggingface-cli upload Nanthasit/sakthai-context-7b-merged sakthai-7b-q4_k_m.gguf .
```

## Performance expectations
| Model | CPU (4 cores) | CPU (8 cores) | GPU (T4) |
|---|---|---|---|
| 0.5B Q4 | ~15 tok/s | ~25 tok/s | ~50 tok/s |
| 1.5B Q4 | ~6 tok/s | ~12 tok/s | ~30 tok/s |
| 7B Q4 | ~1 tok/s | ~3 tok/s | ~15 tok/s |
