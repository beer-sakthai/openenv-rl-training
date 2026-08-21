---
name: workflow
description: Use when the user mentions workflow, pipeline, cycle, lifecycle, CI/CD, release process, or end-to-end development of SakThai models. Orchestrates all skills and commands into the 11-phase development cycle. Phase: ALL (master cycle).
---

# SakThai Development Cycle

```
                     ┌──────────────────────────────────────┐
                     │         SAKTHAI FULL CYCLE            │
                     │   (25 skills · 23 commands)           │
                     └──────────────────────────────────────┘

     ╔══════════════════════════════════════════════════════╗
     ║              ┌──────────┐                            ║
     ║              │ 1. DATA  │                            ║
     ║              │ Augment  │                            ║
     ║              │ Version  │                            ║
     ║              └────┬─────┘                            ║
     ║                   ▼                                  ║
     ║              ┌──────────┐     ┌──────────────┐       ║
     ║              │ 2. TRAIN │────▶│   CHEAPEST    │       ║
     ║              │ QLoRA    │     │ Kaggle T4=$0  │       ║
     ║              │ rsLoRA   │     │ HF CPU=$0.01  │       ║
     ║              └────┬─────┘     └──────────────┘       ║
     ║                   ▼                                  ║
     ║     ╔══════════════════════════════════════╗          ║
     ║     ║      3. EVAL (two-tier)              ║          ║
     ║     ║  ┌──────────┐  ┌────────────────┐    ║          ║
     ║     ║  │ BFCL     │  │ BenchmarkQED    │    ║          ║
     ║     ║  │ bench-v2 │  │ AutoQ → AutoE   │    ║          ║
     ║     ║  │ 500 rows │  │ LLM-as-judge    │    ║          ║
     ║     ║  └────┬─────┘  └───────┬────────┘    ║          ║
     ║     ╚═══════╪════════════════╪══════════════╝          ║
     ║             └──────┬─────────┘                         ║
     ║                    ▼                                   ║
     ║     ┌──────────────────────────────────────┐           ║
     ║     │           4. ERROR ANALYSIS           │           ║
     ║     │  Wrong tool? No tool? Hallucinated?   │           ║
     ║     │  Wrong args? Parallel? Context loss?  │           ║
     ║     └────────────────┬─────────────────────┘           ║
     ║                      ▼                                 ║
     ║              ┌──────────────┐                          ║
     ║              │  5. PUBLISH  │                          ║
     ║              │ Merge → GGUF │                          ║
     ║              │ Model card   │                          ║
     ║              │ Limitations  │                          ║
     ║              │ Eval results │                          ║
     ║              └──────┬───────┘                          ║
     ║                     ▼                                  ║
     ║              ┌──────────────┐                          ║
     ║              │  6. DEPLOY   │                          ║
     ║              │ Gradio Space │                          ║
     ║              │ TGI/Endpoints│                          ║
     ║              │ Ollama       │                          ║
     ║              └──────┬───────┘                          ║
     ║                     ▼                                  ║
     ║     ┌──────────────────────────────────────┐           ║
     ║     │   7. DISCOVERABILITY OPTIMIZATION     │           ║
     ║     │  conversational tag → chat widget    │           ║
     ║     │  Inference provider → try in browser │           ║
     ║     │  .eval_results/ → benchmark badges   │           ║
     ║     │  Cross-link models↔datasets↔spaces   │           ║
     ║     │  Merge collections → single entry    │           ║
     ║     └────────────────┬─────────────────────┘           ║
     ║                      ▼                                 ║
     ║     ┌──────────────────────────────────────┐           ║
     ║     │   8. MONITOR + SECURITY               │           ║
     ║     │  Downloads · Likes · Score drift     │           ║
     ║     │  Token audit · Space secrets          │           ║
     ║     └────────────────┬─────────────────────┘           ║
     ║                      ▼                                 ║
     ║              ┌──────────────┐     ┌───────────────┐    ║
     ║              │  9. PROFILE  │     │ 10. TROUBLE-   │    ║
     ║              │ tok/s bench  │     │    SHOOT       │    ║
     ║              │ vLLM deploy  │     │ Fix errors     │    ║
     ║              └──────┬───────┘     └───────┬───────┘    ║
     ║                     └──────┬──────────────┘            ║
     ║                            ▼                           ║
     ║              ┌─────────────────────────┐               ║
     ║              │   11. ITERATE (or loop)  │              ║
     ║              │  Use insights → re-run   │              ║
     ║              │  5 iterations to target  │              ║
     ║              │  0.5B's 91.2% score      │              ║
     ║              └─────────────────────────┘               ║
     ╚══════════════════════════════════════════════════════════╝
```

## Phase 1 — DATA

**Goal**: Prepare, improve, and version datasets for training and evaluation.

| Task | Command/Skill | Description |
|---|---|---|
| Check existing datasets | `/hf-explore` | Inventory all 8 datasets |
| Update dataset card | `/hf-dataset` | Add card with format, splits, license |
| Add new examples | `data` skill | Follow v7 format (JSONL, tool_call) |
| Generate synthetic data | `/hf-augment` | Hard negatives, irrelevance, multi-hop |
| Create new version | `migration` skill | v7 → v8, breaking changes log |
| Push new version | `hub-api` skill | `api.upload_folder()` or `push_to_hub` |

**Decision gate**: Is the data balanced? (simple/parallel/irrelevance) → Yes → Phase 2. No → fix splits.

**Current state**: v7 has 2,424 rows. v6 has 175 downloads, v7 has 0 — cross-links still point to v6.

## Phase 2 — TRAIN

**Goal**: Fine-tune a SakThai model with QLoRA + rsLoRA. Cheapest path: Kaggle T4 (free) for training.

| Task | Command/Skill | Description |
|---|---|---|
| Choose model size | `training` skill | 0.5B (CPU), 1.5B (Kaggle T4), 7B (HF Jobs T4) |
| Choose hardware | `cost-optimization` skill | Kaggle=$0, HF Jobs T4=$0.40/hr, CPU=$0.01/hr |
| Prepare script | `training` skill | Copy `train-sakthai-1.5b-v2.py`, adjust params |
| Launch training | `training` skill | Kaggle notebook or `hf jobs uv run ...` |
| Verify push | `training` skill | Check adapter + merged repos on Hub |

**Current best config** (from v2 adapter_config.json):
- rsLoRA: enabled, Target modules: all 7 linear
- Rank: 16, Alpha: 32, Dropout: 0.05
- Data: sakthai-combined-v7 + irrelevance-supplement

**Decision gate**: Did training complete without error? → Yes → Phase 3. No → `/hf-troubleshoot`.

## Phase 3 — EVAL

**Goal**: Two-tier evaluation: BFCL (fast) + BenchmarkQED (deep). Run on HF Jobs CPU ($0.01/hr).

| Task | Command/Skill | Description |
|---|---|---|
| Run BFCL eval | `/hf-eval` | `eval_sakthai_15b_v2_fixed.py` against bench-v2 |
| Run cross-model comparison | `eval` skill | Compare 0.5B/1.5B/7B on same benchmark |
| Run BenchmarkQED | `/hf-bench` | AutoQ → model inference → AutoE pairwise |
| Analyze errors | `/hf-analyze-errors` | Classify failures → targeted fixes |

**Critical benchmark data** (current state):
| Model | Selection | Arguments | Irrelevance |
|---|---|---|---|
| 0.5B-merged | **91.2%** | 45.7% | 93.3% |
| 1.5B-merged | **48.2%** | — | — |
| 7B-merged | **57.0%** | — | — |

The 0.5B outperforms larger models because it uses **all linear modules** as LoRA targets + **prompt-masked loss**, while 1.5B/7B only use q/k/v/o. Fix this in Phase 2.

**Decision gate**: Does new model outperform previous? → Yes → Phase 4. No → Phase 4 (error analysis).

## Phase 4 — ERROR ANALYSIS

**Goal**: Understand what the model got wrong and plan targeted fixes.

| Error type | Prevalence | Fix | Effort |
|---|---|---|---|
| Wrong tool selection | ~30% of errors | Add hard negatives, similar-tool pairs | Medium |
| No tool called | ~25% | Expand irrelevance supplement | Low |
| Hallucinated tool | ~15% | Enforce tools param in training | Low |
| Wrong arguments | ~20% | Schema validation during data gen | Medium |
| Parallel call errors | ~10% | Multi-call examples | Medium |

```python
/hf-analyze-errors  # Deep-dive with confusion matrix
/hf-augment         # Generate targeted training data for fix
```

**Decision gate**: Error root cause identified? → Yes → back to Phase 1 with fix. No → `/hf-troubleshoot`.

## Phase 5 — PUBLISH

**Goal**: Push model, create GGUF, write complete model card, update collection.

| Task | Command/Skill | Description |
|---|---|---|
| Merge LoRA → full weights | `/hf-publish` | `merge_and_unload()`, upload model.safetensors |
| Create GGUF | `/hf-quantize` | llama.cpp convert + quantize (Q4_K_M, Q5_K_M) |
| Write model card | `/hf-publish` | Description, usage, eval table, limitations |
| Add .eval_results/ | `hub-api` skill | New format (replaces old model-index YAML) |
| Update collection | `hub-api` skill | Add to `sakthai-model-family` |

**Model card checklist** (every card MUST have):
- [ ] `conversational` tag (enables chat widget)
- [ ] `datasets:` points to v7 (not v6)
- [ ] `widget:` 3-5 examples with outputs
- [ ] `model-index` or `.eval_results/` with bench-v2 scores
- [ ] Usage code (transformers + PEFT + Ollama)
- [ ] Limitations section
- [ ] License + license_link
- [ ] Cross-links to sibling models, datasets, Spaces

## Phase 6 — DEPLOY

**Goal**: Make the model usable via Spaces, Inference Endpoints, Ollama.

| Task | Command/Skill | Hardware | Cost |
|---|---|---|---|
| Build tool-calling demo | `/hf-space` | CPU Basic (0.5B) | Free (PRO) |
| Build vision demo | `/hf-space` | T4 (apply Community Grant) | Free or $0.40/hr |
| Build TTS demo | `/hf-space` | CPU Basic | Free (PRO) |
| Inference Endpoint | `/hf-deploy` | T4 (1.5B) | $0.40/hr |
| Ollama integration | `/hf-ollama` | Local | Free |

**Upgrade path**: All 3 existing Spaces are **static HTML** → convert to Gradio. 0.5B fits free CPU tier.

## Phase 7 — DISCOVERABILITY OPTIMIZATION

**Goal**: Get models found, tried, and shared. This is where the ecosystem grows.

| Action | Impact | Effort |
|---|---|---|
| Add `conversational` tag to all text-gen models | Chat widget on page | 5 min |
| Get 0.5B on Featherless AI inference provider | 1,370 visitors can try it | 1 email |
| Add `.eval_results/` format | Benchmark badges on page | 30 min |
| Fix all cross-links to point to v7 (not v6) | Dataset downloads | 15 min |
| Merge 2 collections into 1 | Clean navigation | 10 min |
| Add Limitations section to every card | Credibility | 30 min |

## Phase 8 — MONITOR + SECURITY

**Goal**: Track health, catch drift, keep tokens safe.

| Task | Command/Skill | How often |
|---|---|---|
| Check downloads/likes | `/hf-monitor` | Weekly |
| Re-run bench-v2 for drift | `/hf-eval` | Monthly |
| Audit tokens/credentials | `/hf-security` | Monthly |
| Check Space logs | HF Space dashboard | Bi-weekly |

## Phase 9 — PROFILE

**Goal**: Optimize inference speed for production.

| Scenario | Setup | Expected tok/s |
|---|---|---|
| 0.5B GGUF on laptop CPU | Ollama | ~15 tok/s |
| 1.5B BF16 on T4 | transformers | ~30 tok/s |
| 1.5B on T4 with vLLM | vLLM server | ~80 tok/s |
| 7B on A10G | TGI | ~40 tok/s |

## Phase 10 — TROUBLESHOOT

**Goal**: Fix errors fast. See `/hf-troubleshoot` for 15+ common issues with exact fixes.

## Phase 11 — ITERATE (5-LOOP)

**Goal**: Close the gap to 0.5B's 91.2% through progressive improvement.

| Iter | Focus | Data change | Target |
|---|---|---|---|
| 1 | Merge v2 LoRA → benchmark | None (already trained) | > 48.2% |
| 2 | Retrain with rsLoRA + all 7 modules | Same data | > Iteration 1 |
| 3 | Pure tool-calling data (no chat) | Filtered dataset | > 70% |
| 4 | Data augmentation + hard negatives | Augmented dataset | > 85% |
| 5 | Production polish | Best config | > Iteration 4 |

**Cost for all 5 iterations**: ~$0.02 (Kaggle T4 for training, HF Jobs CPU for eval)

## All commands by phase

| Phase | Commands |
|---|---|
| DATA | `/hf-explore`, `/hf-dataset`, `/hf-augment` |
| TRAIN | (scripts: `train-sakthai-1.5b-v2.py`), `/hf-cost` |
| EVAL | `/hf-eval`, `/hf-bench`, `/hf-analyze-errors` |
| PUBLISH | `/hf-publish`, `/hf-quantize` |
| DEPLOY | `/hf-space`, `/hf-deploy`, `/hf-agent`, `/hf-ollama` |
| DISCOVER | `/hf-discover` |
| MONITOR | `/hf-monitor`, `/hf-security` |
| PROFILE | `/hf-profile` |
| FIX | `/hf-troubleshoot` |
| PLAN | `/hf-plan`, `/hf-cycle`, `/hf-loop` |
| MIGRATE | `/hf-migrate` |
| CONTRIBUTE | `/hf-contribute` |
