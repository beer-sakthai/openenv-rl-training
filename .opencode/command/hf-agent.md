---
description: Build a smolagents agent using any SakThai model (0.5B CPU, 1.5B/7B API) for tool-calling tasks. Generate runnable scripts.
agent: general
---

Help build a smolagents agent using the SakThai model family.

1. Read the smolagents skill for reference code.
2. Three modes based on hardware:

   **Mode A — 0.5B local CPU**
   - Model: `Nanthasit/sakthai-context-0.5b-merged`
   - `TransformersModel(device="cpu")`
   - Works on any machine

   **Mode B — 1.5B local GPU**
   - Model: `Nanthasit/sakthai-context-1.5b-merged`
   - `TransformersModel(device_map="auto")`
   - Needs GPU (Kaggle T4, HF Space, etc.)

   **Mode C — Any model via HF API**
   - `HfApiModel(model_id="Nanthasit/sakthai-context-1.5b-merged", token=HF_TOKEN)`
   - No GPU needed, fastest option

3. Generate a complete script with:
   - Tool definitions (weather, calculator, etc.)
   - Model loading in chosen mode
   - Agent creation (CodeAgent or ToolCallingAgent)
   - Sample query execution
   - Result display

4. For agent demos, suggest pairing with Gradio (`/hf-space`) to create an interactive agent Space.
