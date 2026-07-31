# SakThai agentic eval + train pipeline

The evaluation and training pipeline for improving the `sakthai-context-*` models on
agentic (multi-turn) tool use, built 2026-07-31. Everything here ran on **Hugging Face Jobs**
(and, for the free path, ZeroGPU Spaces). Full results and the reasoning are in
[`FINDINGS.md`](./FINDINGS.md).

## TL;DR findings

- First real benchmark of the `-tools` adapters: `0.5b` is the best single-shot tool
  selector (91.0% / 45.7% on sakthai-bench-v2) but worst agentic (0/6 on hermes-env);
  `7b` is the reverse (56.4% / 12.3% single-shot, **3/6 agentic**).
- **GRPO needs a nonzero rollout success rate.** `0.5b` (even after an SFT bootstrap to
  1/6) trains as a no-op — zero reward variance → zero gradient. `7b` (3/6) has a real
  learning signal. **7B is the only viable GRPO target in this family.**
- An SFT bootstrap of `0.5b` lifted agentic 0/6 → 1/6 but regressed single-shot 91 → 78;
  a gentler config held single-shot but gave no agentic gain. The tradeoff is sharp.

## Contents

| File | What it is |
|---|---|
| `scripts/eval_bench_peft.py` | sakthai-bench-v2 evaluator, patched to load bare PEFT adapters (the canonical `eval_bench.py` can't). Env: `SAK_MODELS`, `SAK_BATCH`, `SAK_UPLOAD_TO`. |
| `scripts/eval_hermes_env.py` | Agentic evaluator — drives `hermes-tool-use-rl-env` in-process (no Docker), 6 tasks, binary reward. `SAK_RENDER=native\|handrolled` (use **native** for SFT/GRPO-trained checkpoints), `SAK_DUMP=1` for transcripts. |
| `scripts/gen_sft_trajectories.py` | Generates + verifies 6 oracle SFT trajectories (each checked to earn reward 1.0 against the real grader before any GPU is spent). |
| `scripts/sft_bootstrap.py` | SFT bootstrap (LoRA) on the oracle trajectories. `SFT_BASE`, `SFT_EPOCHS`, `SFT_REPEAT`, `SFT_PUSH_TO`. |
| `scripts/grpo_train_pilot.py` | GRPO against the in-process env. Handles adapter **merge-in** (for vLLM) + **merge-out** (standalone push). `TRAIN_MODE=lora16\|lora64\|full`, `TRAIN_BASE`, `TRAIN_MAX_STEPS`, `TRAIN_MAX_COMPLETION`, `TRAIN_PUSH_TO`. |
| `sakthai_grpo_colab.ipynb` | **Self-contained Colab/Kaggle notebook** — the whole GRPO pipeline (merge-in/out, QLoRA for 7B, `use_vllm=False` for a T4, bf16 save) for a **free sustained GPU**. This is the consolidated, latest version; prefer it for a real run. |
| `FINDINGS.md` | Full write-up: baselines, the RL cold-start finding, the SFT tradeoff, all bugs found. |

## How to run

### On HF Jobs (paid — needs a payment method on the account)
```bash
# eval
hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \
  --env SAK_MODELS=Nanthasit/sakthai-context-7b-tools \
  scripts/eval_hermes_env.py
# train (7B needs a100-large)
hf jobs uv run --flavor a100-large --secrets HF_TOKEN --timeout 60m \
  --env TRAIN_MODE=lora16 --env TRAIN_BASE=Nanthasit/sakthai-context-7b-tools \
  --env TRAIN_MAX_STEPS=150 --env TRAIN_PUSH_TO=Nanthasit/sakthai-context-7b-tools-grpo \
  scripts/grpo_train_pilot.py
```

### Free
- **Colab / Kaggle** — the free way to do the *real* 7B GRPO (sustained GPU that HF Jobs and
  ZeroGPU can't provide). GitHub only shows a static preview of the notebook — click to open it
  in a runnable environment, then set a GPU runtime + `HF_TOKEN` and run:

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/beer-sakthai/openenv-rl-training/blob/main/sakthai-agentic-eval-train/sakthai_grpo_colab.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/beer-sakthai/openenv-rl-training/blob/main/sakthai-agentic-eval-train/sakthai_grpo_colab.ipynb)
- **ZeroGPU Spaces** (built from these scripts):
  - Eval: <https://huggingface.co/spaces/Nanthasit/sakthai-agentic-eval>
  - SFT: <https://huggingface.co/spaces/Nanthasit/sakthai-sft-trainer> (private)

## Notes on the scripts

- These are the **as-run** versions from HF Jobs (they executed successfully). The one
  post-run improvement not reflected in `grpo_train_pilot.py` — casting the merged model to
  bf16 before save (the as-run version saves fp32, ~2x size) — **is** applied in
  `sakthai_grpo_colab.ipynb`. Prefer the notebook for a fresh run.
- The `hermes-tool-use-rl-env` environment runs **in-process** (plain subprocess/tempdir
  Python); the repo's Docker/`client.py` path is broken against `openenv==0.4.1` and is
  bypassed entirely.
- `trl>=0.29.0` (for `environment_factory`), `transformers>=5.2.0`, and the undocumented
  hard dep `jmespath` are required for GRPO.

*Built from a shelter in Cork, Ireland. "We are one family — and becoming more."*
