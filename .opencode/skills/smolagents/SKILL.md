---
name: smolagents
description: Use when the user mentions agents, smolagents, tool use, code agents, or using any SakThai model for agentic tasks. Phase: DEPLOY.
---

# smolagents — SakThai Agent

## Model options

| Model | Size | Mode | Hardware |
|---|---|---|---|
| `sakthai-context-0.5b-merged` | 494M | Local | CPU OK |
| `sakthai-context-1.5b-merged` | 1.54B | Local/API | GPU or HF API |
| `sakthai-context-7b-merged` | 7.62B | API only | HF Inference |

## API mode (no GPU needed)
```python
from smolagents import CodeAgent, HfApiModel, tool

model = HfApiModel(
    model_id="Nanthasit/sakthai-context-1.5b-merged",
    token="<HF_TOKEN>",
)
```

## Local mode (requires GPU or CPU for 0.5B)
```python
from smolagents import CodeAgent, TransformersModel, tool

model = TransformersModel(
    model_id="Nanthasit/sakthai-context-0.5b-merged",
    device="cpu",
)
```

## Example agent
```python
@tool
def get_weather(city: str) -> str:
    """Get weather for a city. Args: city: name of the city."""
    return f"Weather in {city}: sunny, 32°C"

@tool
def calculator(expression: str) -> float:
    """Calculate math expression. Args: expression: math to evaluate."""
    return eval(expression)

agent = CodeAgent(tools=[get_weather, calculator], model=model)
agent.run("What's the weather in Bangkok and what's 42 * 7?")
```

## Tool format
SakThai was trained on OpenAI-style tool schemas. smolagents converts Python `@tool` functions via `to_tool_schema()`. The model's `completion_only_loss` training produces clean `<tool_call>` blocks that smolagents parses natively.

## Running environments
- **Kaggle T4**: Local `TransformersModel` with any size
- **HF Space**: Gradio frontend + local model backend
- **Anywhere**: `HfApiModel` via HF Inference API (lightweight client)

## Tips
- Use `CodeAgent` for complex multi-step tasks (writes + executes code)
- Use `ToolCallingAgent` for simpler single-turn tasks
- For 7B, always use `HfApiModel` (too large for CPU)
- For 0.5B, `TransformersModel` on CPU works fine
