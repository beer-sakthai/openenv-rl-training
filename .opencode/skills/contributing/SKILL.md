---
name: contributing
description: Use when the user mentions open source, contributions, community, pull requests, issues, forking, or how others can help with the SakThai project. Phase: ALL.
---

# Contributing — SakThai Open Source

## How others can contribute

### 1. Use the models
The simplest contribution: download, try, and give feedback.
```bash
pip install transformers
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Nanthasit/sakthai-context-1.5b-merged")
```

### 2. Report issues
Found a bug? Open an issue on GitHub:
https://github.com/beer-sakthai/Sak-Family-Agent/issues

Good issue reports include:
- Model name and version
- Your code snippet
- Expected vs actual output
- Error message (full traceback)

### 3. Improve data
The training datasets are open. Want to add more tool-calling examples?
1. Fork `sakthai-combined-v7` on HF
2. Add examples in JSONL format
3. Open a PR or contact Beer on GitHub

### 4. Build a demo
The Spaces are currently static HTML. Build an interactive demo:
- Tool-calling chat interface (Gradio)
- Vision demo (Gradio + T4 GPU)
- TTS playground (Gradio, CPU-friendly)

### 5. Translate
Add non-English tool-calling examples:
- Thai (currently ~100 examples)
- Other languages
Contact Beer to add your data to the next dataset version.

### 6. Benchmark
Run the models on your own eval set and share results.
Use https://huggingface.co/datasets/Nanthasit/sakthai-bench-v2

### 7. Financial support
No Patreon, no GitHub sponsors. If the project helps you:
- ⭐ Leave a like on HF models
- 🔁 Share with your network
- 🍴 Fork and build on it

## Contributor guidelines

### Code style
- Python 3.14+, type hints encouraged
- Follow existing patterns in training scripts
- No comments in code (project convention)

### Dataset contributions
Follow the v7 format:
```json
{
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "...", "tool_calls": [...]}],
  "tools": [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]
}
```

### Licensing
All SakThai models and datasets are Apache 2.0.
Contributions are accepted under the same license.

## Project roadmap (community involvement)

| Area | Help needed | Skill level |
|---|---|---|
| Data augmentation | More training examples (especially Thai) | Beginner |
| Demo Spaces | Gradio UI development | Intermediate |
| GGUF quantization | llama.cpp conversion | Intermediate |
| Benchmarking | Run models on various hardware | Beginner |
| Documentation | Improve model cards | Beginner |
| Multi-language | Translate tool-calling data | Any |

## Contact
- **GitHub**: https://github.com/beer-sakthai
- **HF**: https://huggingface.co/Nanthasit
- **House of Sak**: https://house-of-sak.vercel.app
