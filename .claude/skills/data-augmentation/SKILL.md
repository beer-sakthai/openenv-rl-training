---
name: data-augmentation
description: Techniques and recipes for synthesizing, filtering, and augmenting agentic and tool-calling datasets for SakThai and OpenEnv models. Use when generating training samples, expanding irrelevance coverage, synthesizing hard negatives, constructing multi-hop tool chains, or formatting datasets for Hugging Face Hub publication.
---

# Data Augmentation — SakThai & OpenEnv

## Motivation & Data Lineage

High-performing tool-calling models require training data that penalizes over-calling, handles missing parameters gracefully, and selects exact function signatures among semantically similar tools.

- **Primary dataset series**: `Nanthasit/sakthai-combined-v7` (2,424 rows), updated through `v10` on Hugging Face Hub.
- **Irrelevance supplement**: `Nanthasit/sakthai-irrelevance-supplement` (curated non-tool queries).
- **Core goal**: Balance positive tool selections with negative/irrelevance queries so the model does not hallucinate tool calls on generic conversation.

## Augmentation Strategies

### 1. Hard Negatives (High ROI)
Create scenarios where multiple tools share overlapping domain terminology or identical parameter keys:
- Distinguish `get_weather(city)` vs `get_time(city)`.
- Distinguish `search_products(query)` vs `search_reviews(query)`.
- Distinguish `create_invoice(customer_id, amount)` vs `refund_payment(customer_id, amount)`.

### 2. Irrelevance Expansion
Include queries where NO tool should be invoked (expected `tool_calls: []` or plain assistant text):
- Conversational greetings ("Hello", "Good morning").
- Chitchat and general knowledge queries not requiring API lookups ("What is the capital of France?").
- Out-of-scope requests where available tools cannot satisfy the prompt ("Please order pizza").
- Ambiguous user inputs requiring clarification before calling tools.

### 3. Multi-Hop & Data-Linked Chains
Sequential or interdependent tool calls:
- Query 1 retrieves data -> Query 2 consumes attributes from Query 1 output.
- Example: `find_restaurant(city="Bangkok")` -> `check_table_availability(restaurant_id="123", time="19:00")`.

### 4. Parameter Edge Cases & Normalization
- Empty strings, whitespace variations, and missing optional parameters.
- Type coercion edge cases: integers passed as strings, floats formatted with commas, boolean values.
- Multilingual and non-ASCII inputs: Thai Unicode strings, accented characters, transliterations.

### 5. Round-Trip Verification & Cleaning
When using LLMs (e.g. GPT-4o / Claude) for synthetic generation:
- Validate that the assistant tool call JSON parses without syntax errors.
- Confirm that tool names and parameter keys match the supplied schema exactly.
- Normalize `None` values to empty lists (`tools: []`, `tool_calls: []`) to ensure pyarrow / Hugging Face Datasets compatibility.

## Dataset Structure Reference

Standard OpenAI / TRL chat-template format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant with access to tools."},
    {"role": "user", "content": "What is the temperature in Chiang Mai right now?"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"city\": \"Chiang Mai\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "name": "get_weather",
      "content": "{\"temperature\": 28, \"condition\": \"Partly Cloudy\"}"
    },
    {
      "role": "assistant",
      "content": "The temperature in Chiang Mai is currently 28°C and partly cloudy."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Fetch current weather for a specified location.",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "The city name"}
          },
          "required": ["city"]
        }
      }
    }
  ]
}
```

## Publishing Augmented Datasets to Hub

```python
from datasets import Dataset, concatenate_datasets, load_dataset

# Load existing base and combine with new verified rows
existing_ds = load_dataset("Nanthasit/sakthai-combined-v10", split="train")
augmented_ds = Dataset.from_list(new_rows)
combined = concatenate_datasets([existing_ds, augmented_ds])

# Push to Hub with explicit train split
combined.push_to_hub("Nanthasit/sakthai-combined-v11", split="train")
```
