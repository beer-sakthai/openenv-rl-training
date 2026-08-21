---
description: Check download stats, likes, engagement, and eval score drift across all SakThai models, datasets, and spaces.
agent: general
---

Run a health check on the SakThai ecosystem. Read the monitoring skill for reference.

1. Fetch all models, datasets, spaces via HF Hub API.
2. Report:
   - **Models**: downloads, likes, last modified — highlight any drops
   - **Datasets**: downloads — highlight zeros (v7, bench-v2, irrelevance)
   - **Spaces**: status — are any failing?
   - **Collections**: item count — are new repos included?
3. Compare vs previous check (if data available).
4. Flag any concerns:
   - Downloads stagnating for 2+ weeks
   - Zero likes on flagship models
   - Spaces showing errors
   - Eval scores out of date (last bench-v2 run > 30 days)
5. Suggest specific actions for each flag.

Return a structured report with a status table and recommendations.
