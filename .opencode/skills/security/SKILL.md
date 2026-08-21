---
name: security
description: Use when the user mentions security, secrets, tokens, HF_TOKEN, API keys, permissions, gated repos, or safe sharing of models/datasets. Phase: MONITOR.
---

# Security — SakThai Best Practices

## Token management

### HF_TOKEN (critical!)
Used for: pushing models/datasets, running HF Jobs, accessing gated repos.
```bash
# Set in environment (Windows PowerShell)
$env:HF_TOKEN = "hf_..."

# Pass to HF Jobs
hf jobs uv run --secrets HF_TOKEN train-sakthai-1.5b-v2.py

# Kaggle: Add as notebook secret (click "Add-ons" → "Secrets")
```

**NEVER** commit HF_TOKEN to Git or include in Python scripts.

### Token scopes
Create tokens with minimum permissions:
- **Write**: For pushing models, datasets, Spaces
- **Read**: For inference, downloading
- **Fine-grained**: Limit to specific repos

https://huggingface.co/settings/tokens

## Gated repos

### Making a dataset gated
```python
api.update_repo_settings(
    repo_id="Nanthasit/sakthai-combined-v7",
    private=False,
    gated="manual",  # users must agree to terms
)
```
The `SimpleToolCalling` dataset uses `gated: auto` (requires login). Use `manual` to require clicking through terms.

## Secrets for Spaces
```python
from huggingface_hub import HfApi
api = HfApi()
api.add_space_secret(
    repo_id="Nanthasit/sakthai-demo",
    key="HF_TOKEN",
    value="hf_...",
)
```
Secrets are encrypted and cannot be read after setting. They're available as environment variables inside the Space.

## Safe model sharing

### What to check before publishing
- [ ] No hardcoded API keys in scripts or configs
- [ ] No personal information in training data
- [ ] Tokenizer doesn't expose unexpected special tokens
- [ ] License is set correctly (Apache-2.0 for SakThai)
- [ ] For vision/TTS: no embedded PII in demo images/audio

### What to include in model card
- "Intended Use" section
- "Limitations" section (see huggingface/fixes template)
- "Biases" acknowledgement

## Kaggle security
- Use Kaggle Secrets for HF_TOKEN, not hardcoded values
- Notebooks are public by default — check before publishing
- Remove any print() statements that output tokens or keys

## Emergency: leaked token
If a token is accidentally exposed:
1. Go to https://huggingface.co/settings/tokens
2. Delete the compromised token
3. Create a new one
4. Check Git history: `git log --all -p | grep "hf_"`
