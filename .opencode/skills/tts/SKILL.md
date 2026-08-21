---
name: tts
description: Use when the user mentions TTS, text-to-speech, Kokoro, sakthai-tts-model, voice, or audio generation with the SakThai ecosystem. Phase: DEPLOY.
---

# SakThai TTS — sakthai-tts-model

## Model info
- **Repo**: `Nanthasit/sakthai-tts-model`
- **Base**: Kokoro-82M
- **Size**: 82M params
- **Downloads**: 150
- **Languages**: 15 languages
- **Type**: text-to-speech

## Usage
Kokoro is CPU-friendly (82M params). The model card should include usage code.

## Space
Existing `sakthai-tts` Space is **static HTML** only. Kokoro-82M is small enough to run on a free CPU Space as a Gradio app.

## Recommendations
- Convert static Space to Gradio with live TTS inference
- Add a language selector and voice speed controls
- Publish ONNX export for faster CPU inference
- Add usage code snippet to model card:
```python
# Usage depends on Kokoro implementation
# See https://huggingface.co/Nanthasit/sakthai-tts-model
```
