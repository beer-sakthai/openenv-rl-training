---
description: Publish any SakThai model (adapter, merged, or mergekit) with model card, metadata, and GGUF. Handles 0.5B, 1.5B, and 7B.
agent: general
---

Help publish SakThai models. For each model size:

**0.5B**: `Nanthasit/sakthai-context-0.5b-merged` (has GGUF, card exists)
**1.5B v1**: `Nanthasit/sakthai-context-1.5b-merged` (has GGUF, most downloads)
**1.5B v2** (⚠️ empty): `Nanthasit/sakthai-plus-1.5b-lora` — mergekit merge not run yet. Merge `Nanthasit/sakthai-context-1.5b-merged` + `okawo80085/sakura-tools-1.5b-v1`, then upload.
**7B**: `Nanthasit/sakthai-context-7b-merged` (needs GGUF, card improvements)

For each, craft a model card covering:
- Description and base model
- Training dataset (`sakthai-combined-v7`)
- Method (QLoRA + rsLoRA, completion-only loss)
- Evaluation results (BFCL-style) as `model-index` YAML
- Usage with transformers/PEFT code snippet
- GGUF download link if available
- Link to demo Space and HF Collection

Existing collections (add items, don't recreate):
- `Nanthasit/sakthai-model-family`
- `Nanthasit/sakthai-context-models`
