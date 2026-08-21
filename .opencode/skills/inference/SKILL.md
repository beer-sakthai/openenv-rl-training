---
name: inference
description: Use when the user mentions deployment, TGI, Text Generation Inference, Inference Endpoints, serving, ollama, llama.cpp, or hosting any SakThai model for production. Phase: DEPLOY.
---

# Inference — Serving SakThai Family

## Text Generation Inference (TGI)

### Deploy via Inference Endpoints
1. Go to https://ui.endpoints.huggingface.co/
2. Create new endpoint:

| Model | Instance | Cost/hr |
|---|---|---|
| `sakthai-context-0.5b-merged` | CPU | ~$0.10 |
| `sakthai-context-1.5b-merged` | T4 small | ~$0.60 |
| `sakthai-context-7b-merged` | T4 small | ~$0.60 |

3. Framework: TGI (all sizes)
4. Use the generated API URL

### API usage
```python
import requests
resp = requests.post(API_URL, json={
    "inputs": "<|im_start|>user\nWhat's the weather?<|im_end|>\n<|im_start|>assistant\n",
    "parameters": {"max_new_tokens": 256}
}, headers={"Authorization": f"Bearer {HF_TOKEN}"})
```

### Local TGI with Docker
```bash
docker run --gpus all -p 8080:80 \
  -v $HOME/.cache/huggingface:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id Nanthasit/sakthai-context-1.5b-merged
```

## GGUF inference (llama.cpp / Ollama)

Models with GGUF already published:
- `sakthai-context-0.5b-merged` (Q4_K_M)
- `sakthai-context-1.5b-merged` (Q4_K_M)

Missing: `sakthai-context-7b-merged` — no GGUF yet (most impactful to add)

### llama.cpp usage
```bash
./build/bin/main -m sakthai-context-1.5b-Q4_K_M.gguf \
  -p "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n" \
  -n 256
```

### Ollama Modelfile
```dockerfile
FROM ./sakthai-context-1.5b-Q4_K_M.gguf
TEMPLATE "{{ .Prompt }}"
```
```bash
ollama create sakthai -f Modelfile
ollama run sakthai
```

## Existing inference provider
`sakthai-context-1.5b-merged` is available via **featherless-ai** (inference provider). The 0.5B and 7B models are not yet registered with any provider.

## Hardware requirements
| Model | Size | GPU min | CPU |
|---|---|---|---|
| 0.5B GGUF Q4 | ~300 MB | None | Fast |
| 1.5B GGUF Q4 | ~900 MB | None | OK |
| 7B GGUF Q4 | ~4 GB | T4/RTX 3090 | Slow |
| 7B full bf16 | ~15 GB | A10G | No |
