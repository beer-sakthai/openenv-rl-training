---
name: monitoring
description: Use when the user mentions monitoring, tracking, analytics, downloads, engagement, usage stats, model drift, or wants to check how their SakThai models are performing on HF Hub. Phase: MONITOR.
---

# Monitoring — SakThai Performance Tracking

## Key metrics

| Metric | Source | Current state | Target |
|---|---|---|---|
| Total downloads | HF Hub API | 0.5B: 1,370 / 1.5B: 1,599 / 7B: 744 | Growing trend |
| Total likes | HF Hub API | **1** across all 12 models | > 10 |
| Dataset downloads | HF Hub API | v7: **0** / bench-v2: **0** / irrelevance: **0** | > 100 |
| Space usage | HF Space analytics | No interactive spaces yet | Regular users |
| Eval scores | bench-v2 re-run | 0.5B: 91.2% / 1.5B: 48.2% / 7B: 57.0% | Stable or improving |

## Quick check
```python
from huggingface_hub import HfApi
api = HfApi()

# All models
for m in api.list_models(author="Nanthasit"):
    print(f"{m.modelId:45s} {m.downloads:>6} dl  {m.likes:>3} likes  {m.lastModified:%Y-%m-%d}")

# All datasets
for d in api.list_datasets(author="Nanthasit"):
    print(f"{d.modelId:45s} {d.downloads:>6} dl  {d.lastModified:%Y-%m-%d}")

# All spaces
for s in api.list_spaces(author="Nanthasit"):
    print(f"{s.modelId:45s} {s.lastModified:%Y-%m-%d}")
```

## Score drift detection
Re-run bench-v2 monthly on all merged models to detect drift:
```python
# Run eval_bench.py against all three merged models
models = [
    "Nanthasit/sakthai-context-0.5b-merged",
    "Nanthasit/sakthai-context-1.5b-merged",
    "Nanthasit/sakthai-context-7b-merged",
]
```

## Discoverability fix checklist
- [ ] All models have `conversational` tag (enables chat widget)
- [ ] All models link to `sakthai-combined-v7` in `datasets:` metadata (not just v6)
- [ ] All models have `widget:` with 3-5 diverse examples with outputs
- [ ] v6 is NOT referenced as primary dataset in any model card code snippet
- [ ] Collection items show v7 as primary, not v6
- [ ] At least 1 inference provider per model (currently only 1.5B has one)
- [ ] Spaces are interactive Gradio (not static HTML)
- [ ] Uses `.eval_results/` format instead of old `model-index` YAML

## What to watch for
- **Score drops** — may indicate HF API changes, tokenizer updates, or regressions
- **Download stagnation** — means poor discoverability (fix: cross-links, tutorials, social)
- **Zero likes** — means no community engagement (fix: submit to HF collections, share on social)
- **Space errors** — check Space logs for OOM or dependency issues

## Automated monitoring
Schedule a GitHub Action or HF Space to run daily:
```yaml
name: monitor
on: schedule: [{cron: "0 8 * * 1"}]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - run: pip install huggingface_hub
      - run: python -c "
from huggingface_hub import HfApi
api = HfApi()
for m in api.list_models(author='Nanthasit'):
    print(f'{m.modelId} {m.downloads} {m.likes}')
"
```
