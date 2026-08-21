---
name: migration
description: Use when the user mentions versioning, deprecation, migration, updating models, renaming repos, moving from v1 to v2, or managing breaking changes in the SakThai family. Phase: PUBLISH.
---

# Migration & Versioning — SakThai Model Lifecycle

## Current version map

```
v1 (2026-07-05)
├── sakthai-context-1.5b-tools      ← LoRA adapter (v1 recipe)
├── sakthai-context-1.5b-merged     ← Merged weights + GGUF (flagship)
└── sakthai-context-7b-tools/merged ← Same recipe, 7B scale

v2 (2026-07-30)
├── sakthai-plus-1.5b-lora   ← LoRA adapter (improved: rsLoRA, all targets)
└── sakthai-plus-1.5b  ← NOT YET PUBLISHED (needs merge)
```

## Migration patterns

### User-facing: How to tell users to upgrade
In v1 model card README, add at top:
```markdown
> **⚠️ v2 available**: An improved version is available at
> [sakthai-plus-1.5b-lora](https://huggingface.co/Nanthasit/sakthai-plus-1.5b-lora).
> It uses rsLoRA, targets all 7 linear modules, and trains on the latest dataset.
```

### Internal: Moving from v1 to v2
1. Train v2 with improved recipe
2. Merge LoRA → full weights → upload as `sakthai-plus-1.5b`
3. Add `new_version` metadata to v1 card YAML:
   ```yaml
   new_version: Nanthasit/sakthai-plus-1.5b-lora
   ```
4. Update HF Collection: add v2, keep v1 as "Previous version"

## Repo naming convention
```
sakthai-{domain}-{size}-{variant}[-v{version}]

context-1.5b-merged      ← tool-calling, 1.5B, merged weights, v1 (implicit)
context-1.5b-tools       ← tool-calling, 1.5B, LoRA adapter, v1
context-1.5b-tools-v2    ← tool-calling, 1.5B, LoRA adapter, v2
vision-7b                ← vision, 7B (no variant needed)
```

## Deprecation policy
1. **Mark old repos with deprecation notice** in README top
2. **Keep repos public** — don't delete (breaks existing users' code)
3. **Update collection** — move old items to "Archived" section
4. **Update model card YAML** — add `new_version` pointing to replacement

## Dataset versioning
```
SimpleToolCalling (deprecated) → v6 → v7 → (future) v8
```
Each version should have:
- README noting what changed
- Migration script if format changed
- Old versions remain accessible

## Breaking changes log
| Date | Change | Impact | Migration |
|---|---|---|---|
| 2026-07-30 | v2 uses all 7 linear targets (was 4) | LoRA not compatible with v1 base | Retrain or use merge |
| 2026-07-29 | v7 dataset adds Thai language | New field added (backward-compatible) | None needed |
| 2026-07-10 | v6 → v7 dataset format change | JSONL structure unchanged | None needed |

## Archiving a model
```python
from huggingface_hub import HfApi
api = HfApi()

# Update card with deprecation notice
api.update_model_card(
    repo_id="Nanthasit/sakthai-old-model",
    card_data={"tags": ["deprecated"], "new_version": "Nanthasit/sakthai-new-model"},
)
```
