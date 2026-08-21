---
description: Diagnose and fix common SakThai errors — training crashes, eval issues, HF Jobs failures, tokenizer problems, OOM, and more.
agent: general
---

Help the user diagnose their SakThai issue. Read the troubleshooting skill for reference.

1. Ask what error they're seeing (paste the error message).
2. Classify the error:
   - **Training**: OOM, HF_TOKEN not set, import errors, hanging, NaN loss
   - **Eval**: CUDA not available, garbage output, no tool_call blocks
   - **HF Jobs**: timeout, OOM, missing secrets
   - **Kaggle**: quota exceeded, internet disconnect
   - **Model card**: badges wrong, widget missing
3. Provide the specific fix from the troubleshooting skill.
4. If the error isn't covered, help debug step by step:
   - Check the full traceback
   - Isolate the failing line
   - Check versions (Python, torch, transformers, trl)

Return the diagnosis and fix with exact commands.
