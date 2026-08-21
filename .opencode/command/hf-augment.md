---
description: Generate augmented training data for SakThai — hard negatives, irrelevance expansion, parallel calls, parameter edge cases. Supports Iteration 4 of the learning loop.
agent: general
---

Help the user augment their tool-calling dataset. Read the data-augmentation skill for reference strategies.

1. **Analyze current data**: Fetch `sakthai-combined-v7` to find gaps — which categories are underrepresented? What's the tool coverage?

2. **Choose augmentation type**:
   - Hard negatives: similar tool names/params
   - Irrelevance expansion: when NOT to call tools
   - Multi-hop: chain multiple tools
   - Parameter edge cases: empty, extreme, unicode
   - Parallel calls: 2+ simultaneous tool calls

3. **Generate examples** using an LLM (GPT-4o or similar), following the v7 JSONL format.

4. **Validate**: Check each new example:
   - Gold tool calls exist in the tools list
   - Arguments match the schema
   - No hallucinated tools

5. **Push** as a new dataset split or version:
   ```python
   from datasets import Dataset, load_dataset, concatenate_datasets
   new = Dataset.from_list(new_examples)
   existing = load_dataset("Nanthasit/sakthai-combined-v7", split="train")
   combined = concatenate_datasets([existing, new])
   combined.push_to_hub("Nanthasit/sakthai-combined-v7-augmented", split="train")
   ```

Return a script that generates augmentation examples and pushes to Hub.
