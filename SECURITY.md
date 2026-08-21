# SakThai Security Checklist

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
