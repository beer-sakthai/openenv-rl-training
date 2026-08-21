---
description: Run the full HF platform optimization playbook — fix discoverability, get inference widgets, upgrade Spaces, merge collections, fix dataset cross-links. Turns research into action.
agent: general
---

Apply all findings from the HF platform deep-dive to maximize SakThai's presence. Read the monitoring skill for context.

## Workflow

### 1. Fix metadata on ALL model cards
For every model, ensure:
- `tags:` includes `conversational` (enables chat widget on page)
- `datasets:` points to `sakthai-combined-v7` (not just v6)
- `widget:` has 3-5 diverse examples with expected outputs
- `base_model:` is correct
- `license_link:` added

### 2. Switch default dataset from v6 → v7
All training scripts and model card code snippets reference v6 as the primary dataset. Change to v7:
- Update the `sakthai-combined-v7` dataset card to show working load_dataset() examples
- Change all model card code snippets to use v7
- Update collection item notes to lead with v7, not v6

### 3. Register for free inference widgets
Only `sakthai-context-1.5b-merged` has an inference provider (Featherless AI).
- **0.5B merged** (1,370 downloads, no widget) — highest priority. Contact Featherless AI to also host this model.
- **7B merged** (744 downloads, no widget) — second priority.
- Visit each model page and click "Ask for provider support" to signal demand.

### 4. Merge collections + delete stale one
Two collections exist; "SakThai Context Models" is stale (4 days old) and redundant.
- Merge all items into "SakThai Model Family" (already 27 items)
- Delete or archive "SakThai Context Models"
- Add upvote call-to-action to the main collection description

### 5. Upgrade Spaces from static to Gradio
All 3 spaces are static HTML (no interactivity).
- **sakthai-tts** → Convert to Gradio. Kokoro-82M is CPU-friendly (free tier).
- **sakthai-leaderboard** → Keep static, or upgrade to pull live API stats.
- **sakthai-vision-demo** → Convert to Gradio with T4 GPU. Apply for Community GPU Grant (free).

### 6. Use new .eval_results/ format
Replace old `model-index` YAML with new `.eval_results/sakthai-bench-v2.yaml` files:
```yaml
# .eval_results/sakthai-bench-v2.yaml
task:
  - text-generation
dataset:
  - sakthai-bench-v2
metrics:
  - selection: 91.2
  - arguments: 45.7
  - strict: 45.7
```
Submit to each model repo with `api.upload_file()`.

### 7. Run cheapest 5-iteration loop
| Step | Hardware | Cost |
|---|---|---|
| Training (5x) | Kaggle T4 (free) | $0 |
| Eval (5x) | HF Jobs CPU Basic ($0.01/hr) | ~$0.05 |
| Merge + push | HF Jobs CPU Basic | ~$0.01 |
| **Total** | | **~$0.06** |

Use Kaggle for training (30 hrs/week free quota), HF Jobs CPU for eval (7x cheaper than GPU).

Return a checklist with specific commands and file paths for each action.
