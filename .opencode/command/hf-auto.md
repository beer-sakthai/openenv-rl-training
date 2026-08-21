---
description: Universal auto-command — detects what you need and runs the right action. Checks HF Hub, datasets, training, and model lifecycle automatically.
agent: general
---

Read the user's request and auto-detect which action is needed:

1. **Hub check** — keywords: "check", "status", "downloads", "likes", "explore"
   → Run: list all models/datasets with download counts, likes, last modified
   
2. **Dataset ops** — keywords: "dataset", "v7", "v8", "JSONL", "quality", "combine"
   → Run: load_dataset(), count rows, check format, suggest fixes

3. **Training** — keywords: "train", "fine-tune", "launch", "HF Jobs", "QLoRA"
   → Show: training command, AGENTS.md config, best hardware path

4. **Publish** — keywords: "publish", "upload", "model card", "push", "card"
   → Run: check current card, suggest improvements, push update

5. **Fix** — keywords: "fix", "error", "broken", "issue", "problem"
   → Run: diagnose common issues (Arrow schema, YAML, token, OOM)

6. **Auto-pilot** — default: run full ecosystem health check
   → Run: check all repos status, find gaps, suggest next actions

Return the result of the detected action with exact commands.
