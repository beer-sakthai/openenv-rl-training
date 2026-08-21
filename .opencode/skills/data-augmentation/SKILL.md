---
name: data-augmentation
description: Use when the user mentions data augmentation, synthetic data, hard negatives, data generation, creating training examples, expanding datasets, or improving tool-calling data quality. Phase: DATA.
---

# Data Augmentation — SakThai Tool-Calling

## Why augment
Current v7 dataset has 2,424 rows. The 0.5B model hits 91.2% selection accuracy — but 1.5B/7B lag behind. Augmenting with targeted data can close this gap.

## Augmentation strategies

### 1. Hard negatives (highest ROI)
Examples where tools have similar names or overlapping parameters:
```json
{
  "messages": [
    {"role": "user", "content": "What's the weather in Paris?"}
  ],
  "tools": [
    {"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city", "parameters": {"city": {"type": "string"}}}},
    {"type": "function", "function": {"name": "get_time", "description": "Get time for a city", "parameters": {"city": {"type": "string"}}}}
  ]
}
```
The model must distinguish `get_weather` vs `get_time` when both accept `city`.

### 2. Irrelevance expansion (current supplement has only 60 rows)
Add cases where NO tool should be called:
- Greetings ("Hello!")
- Off-topic ("What's your favorite color?")
- Self-referential ("What can you do?")
- Ambiguous ("Do something")
- Out-of-scope ("Fix my car")

### 3. Multi-hop / data-linked queries
Chain multiple tools:
```json
{"role": "user", "content": "Find restaurants in Bangkok, then get the weather for the top-rated one's city"}
```

### 4. Parameter edge cases
- Empty strings: `{"city": ""}`
- Missing optional params
- Extreme values (very long strings, negative numbers)
- Unicode/Thai characters

### 5. Tool permutation
For N tools, generate examples for all K-choose-N combinations where K is the available tool set. Current v7 has 86 tools — most combos are untested.

### 6. Round-trip consistency
Generate a query → have an LLM produce gold tool calls → verify by executing the tools → keep only verified examples.

## Using LLMs for augmentation
```python
from openai import OpenAI
client = OpenAI()

def augment_example(seed_tools, category):
    """Generate a new training example using GPT-4o."""
    prompt = f"""Generate a tool-calling example. Category: {category}.
    Tools: {json.dumps(seed_tools)}
    Return JSON with 'messages' and 'tools' keys.
    The assistant must call the correct tool(s)."""
    resp = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
    return json.loads(resp.choices[0].message.content)
```

## v7 data structure reference
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [{"function": {"name": "...", "arguments": "..."}}]},
    {"role": "tool", "content": "..."}
  ],
  "tools": [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]
}
```

## Pushing augmented data
```python
from datasets import Dataset, load_dataset, concatenate_datasets
new_ds = Dataset.from_list(new_examples)
existing = load_dataset("Nanthasit/sakthai-combined-v7", split="train")
combined = concatenate_datasets([existing, new_ds])
combined.push_to_hub("Nanthasit/sakthai-combined-v7-augmented", split="train")
```
