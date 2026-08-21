---
description: Create or update dataset cards for all 8 SakThai datasets — v7, v6, irrelevance, bench-v1/v2, SimpleToolCalling, food-penguin, kaggle-notebooks.
agent: general
---

Help create/update dataset cards for the entire SakThai dataset family:

1. **sakthai-combined-v7** (2,424 rows, 0 downloads) — latest training, multilingual en+th
2. **sakthai-combined-v6** (2,116 rows, 175 downloads) — previous version
3. **sakthai-irrelevance-supplement** (60 rows, 0 downloads) — safety data
4. **sakthai-bench-v2** (500 rows, 0 downloads) — official BFCL benchmark
5. **sakthai-bench-v1** (235 rows, 0 downloads) — early benchmark
6. **SimpleToolCalling** (deprecated, gated) — seed data
7. **food-penguin-v1** (648 rows, 51 downloads) — restaurant domain
8. **sakthai-kaggle-notebooks** (103 downloads) — training scripts

For each: check if a card (README.md) already exists via the HF API. If missing or incomplete, craft a card covering: description, how created, format (JSONL), splits, size, license, tags, citation. Push via `HfApi.upload_file`.

Priority order: v7 first (most important, 0 downloads), then bench-v2, then irrelevance supplement.
