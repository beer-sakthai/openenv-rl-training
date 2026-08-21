---
name: autoskill-model
description: AUTO-TRIGGER on model lifecycle operations — create model card, add eval results, update tags, deprecate old versions, cross-link models, update collection. Runs every time you mention model card, publish model, version, deprecate, or cross-link. Phase: PUBLISH (universal).
---

# Autoskill: Model Lifecycle

Auto-triggers on: "model card", "publish model", "version", "deprecate", "eval results", "cross-link", "collection"

## Create model card with eval results
```yaml
---
license: apache-2.0
tags: [sakthai, tool-calling]
datasets: [Nanthasit/sakthai-combined-v7]
base_model: Qwen/Qwen2.5-1.5B-Instruct
---
```

## Add .eval_results/ (benchmark badges)
```python
yaml = "task:\n  - text-generation\ndataset:\n  - sakthai-bench-v2\nmetrics:\n  - selection: 48.2"
api.upload_file(path_or_fileobj=yaml.encode(), path_in_repo=".eval_results/sakthai-bench-v2.yaml",
                repo_id="Nanthasit/sakthai-plus-1.5b")
```

## Cross-link models in README
```markdown
- [v1 (flagship)](https://huggingface.co/Nanthasit/sakthai-context-1.5b-merged)
- [Plus (improved)](https://huggingface.co/Nanthasit/sakthai-plus-1.5b)
- [Plus LoRA](https://huggingface.co/Nanthasit/sakthai-plus-1.5b-lora)
```

## Update collection
```python
api.add_collection_item("Nanthasit/sakthai-model-family",
    item_id="Nanthasit/sakthai-plus-1.5b", item_type="model")
```
