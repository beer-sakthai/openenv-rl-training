---
name: vision
description: Use when the user mentions vision, image-to-text, LLaVA, sakthai-vision-7b, multimodal, or visual question answering with the SakThai ecosystem. Phase: DEPLOY.
---

# SakThai Vision — sakthai-vision-7b

## Model info
- **Repo**: `Nanthasit/sakthai-vision-7b`
- **Base**: LLaVA-1.5-7B
- **Format**: GGUF (llama.cpp compatible)
- **Size**: 6.74B params / ~4 GB (Q4 quant)
- **Downloads**: 186
- **Type**: image-to-text / visual Q&A

## Usage with llama.cpp
```bash
./build/bin/llama-llava-cli \
  -m sakthai-vision-7b.gguf \
  --mmproj mmproj-model.gguf \
  --image photo.jpg \
  -p "Describe this image in detail" \
  -n 200
```

## Usage with transformers
```python
from transformers import LlavaForConditionalGeneration, LlavaProcessor
import torch

model = LlavaForConditionalGeneration.from_pretrained(
    "Nanthasit/sakthai-vision-7b",
    torch_dtype=torch.float16,
    device_map="auto",
)
processor = LlavaProcessor.from_pretrained("Nanthasit/sakthai-vision-7b")
```

## Space
Existing `sakthai-vision-demo` Space is **static HTML** only. Should be upgraded to Gradio with T4 GPU for interactive use.

## Recommendations
- Convert the static Space to Gradio with llama.cpp bindings
- Publish additional quant levels (Q5_K_M, Q8_0)
- Add usage examples to model card
