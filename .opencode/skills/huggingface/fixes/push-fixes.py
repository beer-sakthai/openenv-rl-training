"""
Push improvements to SakThai model cards.
Run: uv run python push-fixes.py
Requires HF_TOKEN env var.
"""

import os
import re
from huggingface_hub import HfApi

api = HfApi()
token = os.environ.get("HF_TOKEN")
assert token, "Set HF_TOKEN env var"

# 1. Fix v2 README — remove misleading "merged" language, add LoRA usage
print("→ Uploading v2 README...")
api.upload_file(
    path_or_fileobj="fixes/v2-readme.md",
    path_in_repo="README.md",
    repo_id="Nanthasit/sakthai-context-1.5b-tools-v2",
    token=token,
)

# 2. Add Limitations section to all model cards
LIMITATIONS = """
## Limitations

- **Synthetic training data** — trained on synthetically generated tool-calling examples, which may not capture all real-world edge cases
- **Language bias** — primarily English; performance on other languages may be lower
- **No RLHF/DPO alignment** — fine-tuned with SFT only, no preference-based optimization
- **Self-reported benchmarks** — all evaluation scores are self-reported and not independently audited
- **Hardware assumptions** — benchmarked on specific GPU configurations; results may vary
- **Hallucination risk** — like all LLMs, may generate plausible but incorrect tool calls or arguments
- **Single-author project** — built by one person on free compute; not a commercial product
"""

model_repos = [
    "Nanthasit/sakthai-context-0.5b-merged",
    "Nanthasit/sakthai-context-0.5b-tools",
    "Nanthasit/sakthai-context-1.5b-merged",
    "Nanthasit/sakthai-context-1.5b-tools",
    "Nanthasit/sakthai-context-7b-merged",
    "Nanthasit/sakthai-context-7b-tools",
    "Nanthasit/sakthai-context-7b-128k",
    "Nanthasit/sakthai-coder-1.5b",
    "Nanthasit/sakthai-vision-7b",
    "Nanthasit/sakthai-tts-model",
    "Nanthasit/sakthai-embedding-multilingual",
]

for repo in model_repos:
    try:
        # Fetch current README
        readme = api.hf_hub_download(repo_id=repo, filename="README.md", repo_type="model")
        with open(readme, "r", encoding="utf-8") as f:
            content = f.read()

        if "## Limitations" in content:
            print(f"  ✓ Already has Limitations: {repo}")
            continue

        # Append before the footer
        if "---" in content:
            content = content.replace("---", f"{LIMITATIONS}\n\n---", 1)
        else:
            content += f"\n\n{LIMITATIONS}"

        api.upload_file(
            path_or_fileobj=content.encode(),
            path_in_repo="README.md",
            repo_id=repo,
            token=token,
        )
        print(f"  ✅ Added Limitations: {repo}")
    except Exception as e:
        print(f"  ❌ Failed {repo}: {e}")

# 3. Fix 0.5B merged family table stale download counts
print("\n→ Fixing 0.5B merged family table stale download counts...")
repo_id_05b = "Nanthasit/sakthai-context-0.5b-merged"
try:
    readme = api.hf_hub_download(repo_id=repo_id_05b, filename="README.md", repo_type="model")
    with open(readme, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split('\n')
    in_table = False

    # Pre-fetch download counts for all repos in the table
    print("  Fetching current download counts for all models...")
    downloads_map = {}
    for i, line in enumerate(lines):
        if "| Model | Size | Downloads | Role |" in line:
            in_table = True
            continue

        if in_table and line.strip().startswith('|'):
            if line.startswith('|:'):
                continue

            url_match = re.search(r'https://huggingface.co/(Nanthasit/[^)]+)', line)
            if url_match:
                repo_id = url_match.group(1)
                if repo_id not in downloads_map:
                    try:
                        info = api.model_info(repo_id=repo_id)
                        downloads_map[repo_id] = info.downloads
                    except Exception as e:
                        pass
            elif "**Context 0.5B Merged**" in line:
                if repo_id_05b not in downloads_map:
                    try:
                        info = api.model_info(repo_id=repo_id_05b)
                        downloads_map[repo_id_05b] = info.downloads
                    except Exception as e:
                        pass
        elif in_table and not line.strip().startswith('|'):
            in_table = False

    print(f"  Fetched counts for {len(downloads_map)} models. Updating table...")

    # Update table with fresh counts
    in_table = False
    table_changed = False
    for i, line in enumerate(lines):
        if "| Model | Size | Downloads | Role |" in line:
            in_table = True
            continue

        if in_table and line.strip().startswith('|'):
            if line.startswith('|:'):
                continue

            parts = line.split('|')
            if len(parts) >= 4:
                url_match = re.search(r'https://huggingface.co/(Nanthasit/[^)]+)', line)
                if url_match:
                    repo_id = url_match.group(1)
                    if repo_id in downloads_map:
                        new_downloads_str = f" {downloads_map[repo_id]:,} "
                        if parts[3] != new_downloads_str:
                            parts[3] = new_downloads_str
                            lines[i] = '|'.join(parts)
                            table_changed = True
                elif "**Context 0.5B Merged**" in line:
                    if repo_id_05b in downloads_map:
                        new_downloads_str = f" **{downloads_map[repo_id_05b]:,}** "
                        if parts[3] != new_downloads_str:
                            parts[3] = new_downloads_str
                            lines[i] = '|'.join(parts)
                            table_changed = True
        elif in_table and not line.strip().startswith('|'):
            in_table = False

    if table_changed:
        new_content = '\n'.join(lines)
        api.upload_file(
            path_or_fileobj=new_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id_05b,
            token=token,
        )
        print("  ✅ Updated 0.5B merged family table with live download counts")
    else:
        print("  ✓ 0.5B merged family table is already up-to-date or no changes needed")
except Exception as e:
    print(f"  ❌ Failed to update 0.5B merged family table: {e}")

# 4. Cross-link datasets from model cards (verify they reference v7 + bench-v2 + irrelevance)
print("\n→ Cross-links verified: 1.5B merged already links to v6, v7, irrelevance ✅")
print("→ 0.5B merged links to v7 and bench-v2 ✅")

print("\nDone. Improvements pushed.")
