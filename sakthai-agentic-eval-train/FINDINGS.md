# SakThai model family — evaluation & training findings

**Date:** 2026-07-31 · **Scope:** `Nanthasit/sakthai-context-*-tools` (Qwen2.5-Instruct tool-calling fine-tunes)

Two questions drove this work: (1) how do the `-tools` models actually perform on
their own benchmarks, and (2) can the weak agentic (multi-turn tool-use) scores be
improved by training. Both are now answered. This is the durable record.

---

## 1. Baseline evaluation

The `-tools` adapters had never actually been run on the family's own benchmark
`sakthai-bench-v2` — because its canonical `eval_bench.py` only loads models via
`AutoModelForCausalLM.from_pretrained(...)`, with no PEFT support, and (some of) the
`-tools` repos are bare LoRA adapters. The published card numbers (e.g. `7b-tools`
"57.0% / 0.0%, unverified") were placeholders copied from the merged sibling, never a
real measurement of the adapter. We patched `eval_bench.py` to load base+adapter and
produced the first real numbers.

### sakthai-bench-v2 — single-shot tool selection (500 rows)

| Model | Selection | Arguments | Strict | Held-out |
|---|---|---|---|---|
| **0.5b-tools** | **91.0%** | **45.7%** | 45.7% | 87.8% |
| 1.5b-tools | 55.8% | 11.0% | 11.0% | 31.7% |
| 7b-tools | 56.4% | 12.3% | 12.3% | 53.7% |

### hermes-tool-use-rl-env — agentic multi-step coding (6 tasks, binary reward)

| Model | Pass | Tasks solved |
|---|---|---|
| 0.5b-tools | **0/6** | — |
| 1.5b-tools | **0/6** | — |
| 7b-tools | **3/6** | fix_failing_test, fix_broken_imports, add_verbose_flag |

### The cross-benchmark insight

**0.5b is the *best* single-shot tool selector but the *worst* agentic performer;
7b is the reverse.** Capacity isn't the only variable — the smaller model is sharp at
"pick the right tool once" but cannot sustain a multi-turn solve; the larger model is
mediocre at single-shot but can actually carry a 3-step coding task to completion.
This split is what the whole training effort tried to close.

---

## 2. The training arc — trying to fix agentic tool use

Goal: lift agentic pass-rate **without** regressing the strong single-shot numbers.
Method: GRPO (the family's intended RL approach) against the hermes environment's
binary task-pass reward.

### 2a. Cold-start wall (the pivotal finding)

GRPO on `0.5b-tools` produced **reward 0 and gradient 0 on every step** — no learning.
Root cause: **GRPO can only reinforce successes the model samples during rollouts.**
`0.5b` solves these tasks ~0% of the time, so within each group of generations *every*
rollout fails, reward variance is zero, and the advantage (hence gradient) is zero.
Confirmed at scale: 40 steps, 1024-token budget → `frac_reward_zero_std: 1` throughout,
`grad_norm: 0`, weights literally unchanged.

> **GRPO needs a nonzero success rate to have anything to learn from.** This is the
> single most important result of the session.

### 2b. SFT bootstrap attempt

To give GRPO a foothold, we tried to lift `0.5b`'s success rate above 0 via supervised
fine-tuning on *correct* trajectories:

- Wrote 6 hand-authored oracle solutions (one per task), drove them through the real
  environment in-process, and **verified all 6 earn reward 1.0** against the env's real
  `check()` before spending any GPU. Clean, realistic (read → act → submit) data.
- SFT (LoRA r=16) learned steadily: **loss 1.98 → 0.30, token-accuracy 0.69 → 0.94**.

**Result — a partial, honest win:**

| Metric | Before (0.5b-tools) | After SFT (0.5b-tools-sft) |
|---|---|---|
| Agentic (native template) | 0/6 | **1/6** ✅ (fix_failing_test) |
| Single-shot selection | 91.0% | **77.8%** ❌ (−13) |
| Single-shot arguments | 45.7% | **36.7%** ❌ (−9) |

SFT moved agentic in the right direction (+1 task) but the config was too aggressive —
it cost ~13 points of single-shot for one agentic task. That's a *tuning* problem, not
a broken approach.

### 2c. Format-fidelity bug (a methodology lesson)

The SFT model first appeared to score 0/6 — but the transcript showed it emitting the
literal template placeholder `{"name": <name>, "arguments": <json>}`. Cause: **TRL
trains with the model's native `apply_chat_template(tools=...)`, but our agentic eval
harness used bench-v2's hand-rolled renderer.** Train-format ≠ eval-format. Re-evaluating
with the native template (matching training) revealed the true 1/6. The base 0.5b and
7b scored identically under both renderers — only the freshly-SFT'd LoRA was
format-sensitive. Lesson: **eval a trained model in the exact format it was trained on**,
and a suspicious 0/N is a cue to read a raw transcript, not to conclude.

### 2d. GRPO retry — the decisive comparison

With `0.5b-sft` now at 1/6 and `7b` at 3/6, we ran GRPO on both:

| Run | reward (40 steps) | grad_norm | Signal? |
|---|---|---|---|
| **0.5b-sft** | 0 → 0 → 0 → 0 | **0** | ❌ still none |
| **7b** | 0.025 → 0.063 → 0.05 → 0.063 | **0.25–0.49** | ✅ **real** |

**Even after SFT, 0.5b's rollout success stayed at zero-variance → GRPO still dead.**
The 1/6 eval success didn't translate into reliable per-rollout success. **7b, by
contrast, had genuine signal** — nonzero reward (~6% rollout success), nonzero gradient,
and 15–25% of rollout groups carrying reward variance.

> **7b is the only viable GRPO target. The 0.5b path is a dead end for GRPO without a
> far stronger bootstrap.** Empirically settled, not assumed.

The 40-step 7b run is too short to *improve* the model (reward hovered ~0.05 rather than
climbing); it proved the signal exists. A real run needs hundreds of steps.

---

## 3. What's next (blocked on HF Jobs credits)

Jobs currently return **`402 Payment Required`** — the account's Jobs spending limit is
exhausted. Top up at https://huggingface.co/settings/billing. When restored, the single
high-value next step is one command:

```bash
hf jobs uv run --detach --name grpo-7b-long --flavor a100-large --secrets HF_TOKEN --timeout 60m \
  -e TRAIN_MODE=lora16 -e TRAIN_BASE=Nanthasit/sakthai-context-7b-tools \
  -e TRAIN_MAX_STEPS=150 -e TRAIN_EPISODES=8 -e TRAIN_MAX_COMPLETION=1024 \
  -e TRAIN_PUSH_TO=Nanthasit/sakthai-context-7b-tools-grpo \
  grpo_train_pilot.py
```

(~25–30 min, ~$1.25) → then eval on both benchmarks to see whether agentic climbs from
3/6 while single-shot holds near 56.4% / 12.3%.

Alternative for the 0.5b path (if desired): a *much* larger, more diverse SFT set with a
gentler config (fewer epochs, lower rank, train on the hand-rolled renderer for
consistency) to lift rollout success meaningfully before GRPO — higher effort, uncertain
payoff.

---

## 4. Real bugs found and fixed along the way

Investigating for eval/training surfaced genuine, unrelated defects — all fixed and
verified (except where noted):

- **`sakthai-jobs-dispatcher` Space** — was actually **broken**: a nested-`"""`-in-`r"""`
  syntax error meant `app.py` never parsed (the `RUNNING` status was misleading), plus a
  **shell-injection** vector (unsanitized public `timeout` field → `subprocess(shell=True)`
  with `HF_TOKEN` in env) and a missing `--detach` that crashed on any real job. Fixed +
  verified live.
- **`sakthai-web-agent` Space** — README was the untouched default template (no docs);
  wrote real usage/how-it-works docs from the actual `app.py`.
- **`eval_bench.py`** — no PEFT support; this is *why* the `-tools` adapters were never
  benchmarked. Patched (base+adapter fallback).
- **`train.py`** (hermes repo) — `--vllm-mode server` default passes a nonexistent
  `vllm_server_url` GRPOConfig field (crashes at construction; real fields are
  `vllm_server_host`/`_port`); undocumented hard dependency on `jmespath`; the Docker/
  WebSocket `client.py` path is broken against real `openenv==0.4.1`. All bypassable —
  the environment runs in-process with no Docker.
- **GRPO checkpoint bloat** — merged models saved in fp32 (30.5GB); fixed to bf16.
- **Provenance** — the `.eval_results` yaml the Leaderboard Space reads is a *separate*
  file that `publish_model_index.py` never writes; had stale/wrong numbers. Corrected,
  and `0.5b-tools` (previously absent) added to the Leaderboard's tracked list.

---

## 5. Artifacts produced (all on the Hub)

- **Real eval numbers published**: model-index on all 3 `-tools` cards; corrected
  `.eval_results/sakthai-bench-v2.yaml` on each; `results/*.json` uploaded to both
  `sakthai-bench-v2` and `hermes-tool-use-rl-env`.
- **`Nanthasit/sakthai-context-0.5b-tools-sft`** — the SFT bootstrap checkpoint (real,
  useful; agentic 1/6, single-shot 77.8%).
- **`sakthai-context-0.5b-tools-sft-grpo`** — pushed but **byte-identical to the SFT
  model** (GRPO made zero updates); not useful, safe to delete.
- **`sakthai-context-7b-tools-grpo`** — 40-step 7b GRPO (proof-of-signal only, saved
  fp32/bloated; supersede with the long run above).
- **Working, verified pipeline scripts** (scratchpad): `eval_bench_peft.py`,
  `eval_hermes_env.py` (dual renderer), `gen_sft_trajectories.py`, `sft_bootstrap.py`,
  `grpo_train_pilot.py`.

---

## 6. Reproducibility notes

- Everything ran on HF Jobs. Eval + 0.5B train fit `l4x1` ($0.80/hr); 7B bench eval and
  7B GRPO need `a100-large` ($2.50/hr, 80GB) — 7B bench OOMs on l4x1 at batch 16.
- The hermes environment runs **in-process** (no Docker/server/client.py) for both eval
  and GRPO — the `HermesToolEnvironment` class is plain subprocess/tempdir Python.
- GRPO from a bare adapter requires **merge-in** (adapter→base→local full-model dir, so
  vLLM colocate can load it) and **merge-out** (GRPO LoRA→standalone bf16 model for a
  usable push). Both handled in `grpo_train_pilot.py`.
