---
description: Audit HF token security, check for exposed secrets, set up Space secrets, review gated repo settings, and enforce best practices.
agent: general
---

Help the user audit and improve security across the SakThai ecosystem.

1. Read the security skill for reference.

2. Run a security audit:
   - Check local files for hardcoded tokens: `Select-String -Pattern "hf_" "*.py"`
   - Verify HF_TOKEN is set as env var, not in scripts
   - Check Space secrets are configured (not hardcoded in app.py)
   - Review gated repo settings for datasets
   - Check Kaggle notebooks for exposed keys

3. If issues found:
   - Rotate any compromised tokens immediately
   - Move tokens to env vars / HF secrets
   - Add `.env` to `.gitignore`
   - Update Kaggle notebooks to use secrets

4. Generate a security checklist and remediation steps.

Return the audit results with specific file paths and commands.
