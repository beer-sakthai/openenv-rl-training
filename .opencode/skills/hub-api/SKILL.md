---
name: hub-api
description: Use when the user mentions huggingface_hub, HfApi, upload, download, create repo, manage models/datasets/spaces programmatically, collections, or Hub operations beyond simple push_to_hub. Phase: ALL.
---

# Hub Python Library — SakThai Operations

## Installation
Already a dependency: `pip install huggingface_hub`

## List all SakThai repos
```python
from huggingface_hub import HfApi
api = HfApi()
models = api.list_models(author="Nanthasit")
datasets = api.list_datasets(author="Nanthasit")
spaces = api.list_spaces(author="Nanthasit")
for m in models: print(f"{m.modelId:40s} {m.downloads:>6} downloads")
```

## Existing collections (already created)
```python
# SakThai Model Family — 25 items
# SakThai Context Models — 8 items
col = api.get_collection("Nanthasit/sakthai-model-family")
for item in col.items:
    print(f"{item.item_type}: {item.item_id}")

# Add to existing collection
api.add_collection_item(
    "Nanthasit/sakthai-model-family",
    item_id="Nanthasit/sakthai-plus-1.5b-lora",
    item_type="model",
)
```

## Common operations

### Create repos
```python
api.create_repo("Nanthasit/sakthai-context-7b-gguf", exist_ok=True)
api.create_repo("Nanthasit/sakthai-demo", repo_type="space", space_sdk="gradio")
api.create_repo("Nanthasit/sakthai-bench-results", repo_type="dataset")
```

### Upload
```python
api.upload_file(path_or_fileobj="README.md", path_in_repo="README.md", repo_id="Nanthasit/sakthai-plus-1.5b-lora")
api.upload_folder(repo_id="Nanthasit/sakthai-demo", repo_type="space", folder_path="./sakthai-demo")
api.upload_large_folder(repo_id="Nanthasit/sakthai-context-7b-gguf", folder_path="./gguf-output/")
```

### Update model card metadata
```python
api.update_model_card(
    repo_id="Nanthasit/sakthai-context-1.5b-merged",
    card_data={
        "tags": ["qwen", "tool-calling", "sakthai", "house-of-sak"],
        "license": "apache-2.0",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    },
)
```

### Manage Space secrets
```python
api.add_space_secret("Nanthasit/sakthai-demo", key="HF_TOKEN", value="hf_...")
```

## Authentication
```python
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])  # or set HF_TOKEN env var
```

## Key reminders
- Collections already exist — add items rather than recreating
- Use `upload_large_folder` for multi-file model uploads (GGUF shards, safetensors)
- Space secrets need re-adding if Space is rebuilt
- Model card metadata (`model-index`) controls how benchmark scores appear on Hub
