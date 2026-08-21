# Security Policy

## Supported Versions

Use this section to tell people about which versions of your project are
currently being supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

Please report (suspected) security vulnerabilities to our team. You can do this by submitting an issue to this repository, clearly marked with "[Security]" in the title.
If the vulnerability is critical, please refrain from sharing sensitive details in the public issue. We will reach out to you directly to establish a secure communication channel.

We aim to acknowledge receipt of vulnerability reports within 48 hours and provide regular updates on the resolution progress.

## Sandboxing Architecture

This repository contains components that execute code in sandboxed environments (`openenv-custom-training/agent_tools/server`). While these environments are designed to isolate execution, they are intended for use in controlled, internal training workflows. Please ensure that untrusted inputs are not passed directly to these environments outside of their intended scope.

---

# SakThai Security Checklist

Operational token-hygiene checklist for anyone driving the HF Jobs / Colab / Kaggle pipeline in this repo.

## Token hygiene

- [ ] `HF_TOKEN` is **never** hardcoded in `.py`, `.ipynb`, `.sh`, `.yaml`, or `.json`
- [ ] All scripts read `HF_TOKEN` from `os.environ["HF_TOKEN"]` (not from a file or inline string)
- [ ] Colab notebooks use `google.colab.userdata.get("HF_TOKEN")` (not `os.environ`)
- [ ] Kaggle notebooks use `from kaggle_secrets import UserSecretsClient` (not hardcoded)
- [ ] HF Jobs use `--secrets HF_TOKEN` flag (not inline token)
- [ ] Secrets are scoped: write tokens only for repos that need writes; read tokens for inference

## Code & config

- [ ] `.gitignore` exists and covers `.env`, `.env.*`, secrets files
- [ ] No API key placeholders in docstrings that could be mistaken for real keys
- [ ] No `print()` or logging statements that output tokens or keys
- [ ] `augment-dataset-10x.py` docstring uses `$OPENAI_API_KEY` not `OPENAI_API_KEY=sk-...`

## Hub repos

- [ ] Gated repos (`sakthai-combined-v7`, `sakthai-bench-v2`) checked: `gated="manual"` or `gated="auto"`
- [ ] Space secrets set via `HfApi().add_space_secret()` not hardcoded in `app.py`
- [ ] Model cards include Limitations section (see `fixes/push-fixes.py`)
- [ ] No PII in training data or demo assets

## Emergency response

If a token is exposed:
1. Go to https://huggingface.co/settings/tokens
2. Delete the compromised token immediately
3. Create a new token with minimal required scopes
4. Update all scripts, Secrets, and CI/CD configs with the new token
5. If committed to Git: `git log --all -p | Select-String "hf_"` — then rotate any repo webhooks
