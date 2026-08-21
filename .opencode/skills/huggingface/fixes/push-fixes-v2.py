"""
Push all critical fixes found during 6-experiment audit.
Run: cd .opencode/skills/huggingface && uv run python fixes/push-fixes-v2.py
Requires HF_TOKEN env var.
"""

import os
from huggingface_hub import HfApi

api = HfApi()
token = os.environ.get("HF_TOKEN")
assert token, "Set HF_TOKEN env var"

# 1. Fix merged-v2 card (auto-generated template, no useful info)
print("→ Uploading merged-v2 model card...")
api.upload_file(
    path_or_fileobj="fixes/merged-v2-card.md",
    path_in_repo="README.md",
    repo_id="Nanthasit/sakthai-context-1.5b-merged-v2",
    token=token,
)

# 2. Add .eval_results/ for merged-v2
print("→ Uploading .eval_results/ for merged-v2...")
api.upload_file(
    path_or_fileobj="fixes/.eval_results/sakthai-bench-v2.yaml",
    path_in_repo=".eval_results/sakthai-bench-v2.yaml",
    repo_id="Nanthasit/sakthai-context-1.5b-merged-v2",
    token=token,
)

# 3. Fix v2 tools README (still auto-generated with licence: license typo)
print("→ Uploading v2 tools README fix...")
api.upload_file(
    path_or_fileobj="fixes/v2-readme.md",
    path_in_repo="README.md",
    repo_id="Nanthasit/sakthai-context-1.5b-tools-v2",
    token=token,
)

print("\n✅ All fixes pushed. Restart opencode for changes.")

print("\n📋 Manual steps remaining:")
print("  1. Run merged-v2 against sakthai-bench-v2 to get actual scores")
print("  2. Update .eval_results/sakthai-bench-v2.yaml with real numbers")
print("  3. Update merged-v2 card with eval results once available")
print("  4. Request 0.5B model on Featherless AI inference provider")
print("  5. Convert static Spaces to Gradio")
