---
description: Deep-dive analysis of eval failures — categorize errors (wrong tool, no tool, hallucinated, wrong args), identify patterns, and suggest targeted data fixes.
agent: general
---

Analyze why a SakThai model is making mistakes on tool-calling eval. Read the error-analysis skill for reference.

1. Fetch the eval results or run a fresh eval on bench-v2.
2. For each incorrect prediction, classify the error:
   - **Wrong tool selection** — called wrong function name
   - **No tool called** — returned text instead of tool_call
   - **Hallucinated tool** — called a tool not in the provided schema
   - **Wrong arguments** — correct tool, incorrect parameters
   - **Parallel call error** — wrong combination of calls
   - **Multi-turn error** — lost context across turns

3. Aggregate by category and tool:
   ```python
   # Which tools are most confused?
   # Which categories (simple/parallel/irrelevance) have highest error rate?
   ```

4. Suggest data fixes:
   - Add hard negatives for confused tool pairs
   - Expand irrelevance supplement for no-tool errors
   - Fix schema validation for argument errors
   - More parallel call examples

Return an error analysis report with a confusion matrix, error distribution, and prioritized fix list.
