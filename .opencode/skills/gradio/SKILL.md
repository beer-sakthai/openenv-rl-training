---
name: gradio
description: Use when the user mentions Gradio, Spaces, demo, UI, web interface, or deploying an interactive demo for any SakThai model (context, vision, TTS). Phase: DEPLOY.
---

# Gradio — SakThai Demo Spaces

## Current state
All 3 existing Spaces (`sakthai-tts`, `sakthai-leaderboard`, `sakthai-vision-demo`) are **static HTML** with no GPU. This skill helps build actual interactive demos.

## Option A: Tool-calling chat Space (flagship)

Model options for different hardware:
| Model | Size | GPU needed | CPU possible? |
|---|---|---|---|
| `sakthai-context-0.5b-merged` | 494M | No | Yes (fast) |
| `sakthai-context-1.5b-merged` | 1.54B | T4 | Yes (slow) |
| `sakthai-context-7b-merged` | 7.62B | T4 | No |

### Example `app.py` (with model selector)
```python
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "0.5B (CPU)": "Nanthasit/sakthai-context-0.5b-merged",
    "1.5B (GPU)": "Nanthasit/sakthai-context-1.5b-merged",
}

def load(model_key):
    repo = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(repo)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype).to(device)
    return tok, m, device

def chat(message, history, model_key, tools_json):
    tok, m, dev = load(model_key)
    msgs = [{"role": "user", "content": message}]
    prompt = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(dev)
    out = m.generate(prompt, max_new_tokens=256)
    return tok.decode(out[0][prompt.shape[1]:], skip_special_tokens=True)

with gr.Blocks(title="SakThai Tool-Calling") as demo:
    gr.Markdown("# SakThai — Tool-Calling Demo")
    model_selector = gr.Dropdown(choices=list(MODELS.keys()), label="Model")
    chatbot = gr.ChatInterface(chat, additional_inputs=[model_selector])

demo.launch()
```

### Creating the Space
```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("Nanthasit/sakthai-demo", repo_type="space", space_sdk="gradio")
api.upload_folder("Nanthasit/sakthai-demo", repo_type="space", folder_path="./sakthai-demo")
```

## Option B: Leaderboard Space (upgrade from static)
Convert `sakthai-leaderboard` from static HTML to Gradio with live data pulled from model-index metadata.

## Hardware tips
- CPU Spaces: 0.5B model only, use float32
- GPU Spaces (T4 small): 1.5B or 7B with bfloat16
- Vision 7B: T4 small with llama.cpp binding
- TTS (Kokoro-82M): CPU-friendly, 82M params
