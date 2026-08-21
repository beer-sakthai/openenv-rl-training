---
description: Deploy any SakThai model (0.5B, 1.5B, or 7B) to HF Inference Endpoints or local TGI. Also covers GGUF inference via llama.cpp/Ollama.
agent: general
---

Help deploy SakThai models for production inference. Options by model:

| Model | Inference Endpoint | Local TGI | GGUF/Ollama |
|---|---|---|---|
| 0.5B merged | CPU ($0.10/hr) | CPU OK | Already exists |
| 1.5B merged | T4 ($0.60/hr) | GPU needed | Already exists |
| 7B merged | T4 ($0.60/hr) | GPU needed | **Missing** |

1. Read the inference skill for reference.
2. **Inference Endpoints** (recommended):
   - Guide: https://ui.endpoints.huggingface.co/
   - Framework: TGI
   - Provide API usage examples with bearer token auth
3. **Local TGI with Docker**:
   ```bash
   docker run --gpus all -p 8080:80 \
     ghcr.io/huggingface/text-generation-inference:latest \
     --model-id Nanthasit/sakthai-context-1.5b-merged
   ```
4. **GGUF inference** (for Ollama/llama.cpp):
   - Show llama.cpp CLI usage
   - Show Ollama Modelfile + `ollama create`
5. Note: 1.5B merged is available via featherless-ai inference provider. 0.5B and 7B are not registered with any provider yet.
