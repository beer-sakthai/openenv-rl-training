---
name: error-analysis
description: Use when the user mentions error analysis, failure analysis, eval failures, why the model got it wrong, error patterns, confusion analysis, or debugging model mistakes in tool-calling. Phase: EVAL.
---

# Error Analysis — SakThai Tool-Calling Failures

## Error categories

### Type 1: Wrong tool selection
Model calls tool B when tool A was correct.
```
Query: "What's the weather in Bangkok?"
Gold:  get_weather({"city": "Bangkok"})
Pred:  get_time({"city": "Bangkok"})
```
**Root cause**: Similar tool names/descriptions, model confused by overlapping schemas.

### Type 2: No tool called (tool omission)
Model responds with text instead of calling a tool.
```
Query: "What's the stock price of Apple?"
Gold:  get_stock_price({"ticker": "AAPL"})
Pred:  "I don't have access to real-time stock data."
```
**Root cause**: Model defaulting to chat behavior, insufficient irrelevance training.

### Type 3: Hallucinated tool
Model calls a tool that doesn't exist in the provided schema.
```
Query: "Send an email"
Gold:  (no tool — no email tool available)
Pred:  send_email({"to": "..."})  ← not in tools list
```
**Root cause**: Model ignoring the tools parameter, generating from pretraining knowledge.

### Type 4: Wrong arguments
Correct tool, wrong parameter values or format.
```
Query: "What's 2+2?"
Gold:  calculator({"expression": "2+2"})
Pred:  calculator({"a": 2, "b": 2})  ← wrong schema
```
**Root cause**: Schema mismatch, model using different parameter names.

### Type 5: Parallel call errors
Correct tool set but wrong combination.
```
Query: "Weather in Bangkok and London"
Gold:  [get_weather({"city": "Bangkok"}), get_weather({"city": "London"})]
Pred:  [get_weather({"city": "Bangkok, London"})]  ← single call, both cities
```
**Root cause**: Model not splitting parallel calls correctly.

### Type 6: Multi-turn context loss
In a conversation, model forgets earlier context or tools.
```
Turn 1: User: "What's the weather?" → Assistant: calls get_weather(...)
Turn 2: User: "And in London?" → Assistant: calls get_time(...)  ← wrong!
```
**Root cause**: Context window issues, training data lacking multi-turn examples.

## Analysis workflow

```python
import json, re
from collections import Counter

def analyze_errors(gold_file, pred_file):
    errors = {"wrong_tool": [], "no_tool": [], "hallucinated": [], "wrong_args": [], "parallel": [], "context": []}
    for gold, pred in zip(gold_file, pred_file):
        g_tools = extract_tools(gold)
        p_tools = extract_tools(pred)
        error_type = classify_error(g_tools, p_tools, gold.get("tools", []))
        errors[error_type].append({"query": gold["messages"][-1]["content"], "gold": g_tools, "pred": p_tools})
    return errors

def classify_error(gold, pred, available_tools):
    available_names = {t["function"]["name"] for t in available_tools}
    pred_names = {c["name"] for c in pred}
    gold_names = {c["name"] for c in gold}

    if not gold_names and pred_names:
        return "hallucinated"  # shouldn't call any tool
    if gold_names and not pred_names:
        return "no_tool"       # should have called a tool
    if pred_names - available_names:
        return "hallucinated"  # called a tool not in the list
    if gold_names != pred_names:
        return "wrong_tool"    # wrong tool selected
    if gold != pred:
        return "wrong_args"    # correct tool, wrong args
    return None  # correct
```

## Current known issues from deep dive

| Issue | Affects | Evidence |
|---|---|---|
| Large models underperform | 1.5B (48.2%), 7B (57.0%) | Chat data dilutes tool-calling |
| No irrelevance supplement used | All v1 models | Supplement has 0 downloads |
| v2 not merged/benchmarked | 1.5B-tools-v2 | Adapter only, no eval |
| No hard negatives | All models | Similar-tool confusion not tested |
| Thai language sparse | v7 dataset | Only ~100 bilingual examples |

## Fix priority by error type
| Error type | Fix | Effort |
|---|---|---|
| Wrong tool selection | Add hard negatives, similar-tool pairs | Medium |
| No tool called | Expand irrelevance supplement | Low |
| Hallucinated tool | Enforce tools parameter in training | Low |
| Wrong arguments | Schema validation during data gen | Medium |
| Parallel call errors | Add multi-call examples | Medium |
| Multi-turn errors | Add more conversational data | High |
