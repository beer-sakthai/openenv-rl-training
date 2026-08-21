---
description: Run BFCL-style evaluation across all SakThai context models (0.5B, 1.5B, 7B) and publish comparable results. Addresses the benchmark inconsistency.
agent: general
---

Help the user run a **cross-model evaluation** to fix the benchmark inconsistency (0.5B at 91.2% vs 1.5B at 48.2%).

1. Read the eval skills and scripts for reference.
2. The key task: run ALL three merged models on the **exact same** `sakthai-bench-v2` with the **exact same scorer**:
   - `Nanthasit/sakthai-context-0.5b-merged`
   - `Nanthasit/sakthai-context-1.5b-merged`
   - `Nanthasit/sakthai-context-7b-merged`
3. Suggest creating a single eval script that iterates all three and produces a unified results table.
4. Execution guide: HF Jobs or Kaggle T4 (no local GPU).
5. After results, publish as `model-index` in each model's README with consistent formatting.
6. Optionally also evaluate `sakthai-plus-1.5b-lora` after mergekit merge is complete.

Metrics to capture: selection accuracy, arguments accuracy, strict accuracy per category (simple/parallel/irrelevance) and overall.
