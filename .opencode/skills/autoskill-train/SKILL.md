---
name: autoskill-train
description: AUTO-TRIGGER on training operations — launch HF Jobs, monitor training, merge LoRA, push adapter, convert to GGUF, run eval. Runs every time you mention train, fine-tune, HF Jobs, QLoRA, or merge. Phase: TRAIN (universal).
---

# Autoskill: Training Operations

Auto-triggers on: "train", "fine-tune", "HF Jobs", "QLoRA", "merge", "GGUF", "eval", "benchmark"

## Launch training
```bash
hf jobs uv run --flavor a10g-small --timeout 6h --secrets HF_TOKEN train-sakthai-1.5b-v2.py
```

## Merge LoRA adapter
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device_map="auto")
merged = PeftModel.from_pretrained(base, "Nanthasit/sakthai-plus-1.5b-lora").merge_and_unload()
merged.push_to_hub("Nanthasit/sakthai-plus-1.5b")
```

## Run eval
```bash
hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \
  --env SAK_MODELS=Nanthasit/sakthai-plus-1.5b \
  --env SAK_BENCH=Nanthasit/sakthai-bench-v2 \
  https://huggingface.co/datasets/Nanthasit/sakthai-bench-v2/resolve/main/eval_bench.py
```

## Create GGUF
```bash
python convert_hf_to_gguf.py ./model --outfile model.gguf
./build/bin/quantize model.gguf model-q4_k_m.gguf q4_K_M
```
