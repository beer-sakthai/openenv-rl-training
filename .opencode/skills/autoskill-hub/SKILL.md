---
name: autoskill-hub
description: AUTO-TRIGGER on any Hugging Face Hub operation — check downloads, upload files, create repos, update cards, manage collections, push datasets, fix metadata. Runs every time you mention HF Hub, model card, upload, push, or repo. Phase: ALL (universal).
---

# Autoskill: HF Hub Operations

Auto-triggers on: "upload", "push to hub", "model card", "dataset card", "repo", "downloads", "collection", "HF Hub", "huggingface hub"

## Universal commands

```python
from huggingface_hub import HfApi
api = HfApi()
```

### Check ecosystem status
```python
for m in api.list_models(author="Nanthasit"):
    print(f"{m.modelId.split('/')[-1]:35s} {m.downloads:>6} dl")
```

### Upload any file
```python
api.upload_file(path_or_fileobj="file.jsonl", path_in_repo="data/file.jsonl",
                repo_id="Nanthasit/sakthai-combined-v8", repo_type="dataset")
```

### Update model card metadata
```python
api.update_model_card_metadata(repo_id="Nanthasit/sakthai-plus-1.5b",
    card_data={"tags": ["sakthai", "plus", "tool-calling"]})
```

### Create repo (auto-detects type)
```python
api.create_repo("Nanthasit/sakthai-new-model", repo_type="model", exist_ok=True)
api.create_repo("Nanthasit/sakthai-new-dataset", repo_type="dataset", exist_ok=True)
```

## Quick links
- https://huggingface.co/Nanthasit
- https://huggingface.co/settings/tokens
