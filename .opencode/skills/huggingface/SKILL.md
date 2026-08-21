---
name: huggingface
description: Use when the user mentions Hugging Face, HF Hub, model publishing, dataset cards, model cards, GGUF, Ollama, or any ecosystem/infrastructure tasks related to the SakThai model family (Nanthasit/* repos). Also triggers on /hf-* commands, push_to_hub, model card, dataset card, BFCL eval, or ecosystem planning. Phase: ALL (master orchestrator).
---

# Hugging Face Ecosystem — SakThai Family

**User**: `Nanthasit`
**Collections**: [SakThai Model Family](https://hf.co/collections/Nanthasit/sakthai-model-family) (25 items), [SakThai Context Models](https://hf.co/collections/Nanthasit/sakthai-context-models) (8 items)

## Models (12 repos)

### Context (tool-calling) models
| Repo | Size | Type | Downloads | Notes |
|---|---|---|---|---|
| `sakthai-context-0.5b-merged` | 494M | Merged + GGUF | 1,370 | Edge/RPi ready |
| `sakthai-context-0.5b-tools` | 494M | Merged | 94 | |
| `sakthai-context-1.5b-merged` | 1.54B | Merged + GGUF | **1,599** | Most popular |
| `sakthai-context-1.5b-tools` | ~66M | LoRA adapter | 349 | v1 adapter |
| `sakthai-plus-1.5b` | 1.54B | Merged (Plus) | 0 | rsLoRA + all 7 targets |
| `sakthai-plus-1.5b-lora` | ~66M | LoRA adapter (Plus) | 0 | rsLoRA + all 7 targets |
| `sakthai-plus-1.5b-coder` | 1.78B | Coder (Plus) | 0 | Code + tool-calling |
| `sakthai-context-7b-merged` | 7.62B | Merged | 744 | No GGUF yet |
| `sakthai-context-7b-tools` | ~52M | LoRA adapter | 399 | |
| `sakthai-context-7b-128k` | — | **Config only (YaRN)** | 506 | No weights |

### Other models
| Repo | Size | Type | Downloads |
|---|---|---|---|
| `sakthai-embedding-multilingual` | 118M | BERT embeddings | 362 |
| `sakthai-vision-7b` | 6.74B | LLaVA GGUF | 186 |
| `sakthai-tts-model` | 82M | Kokoro TTS | 150 |
| `sakthai-coder-1.5b` | 1.78B | Qwen2.5-Coder GGUF | 93 |

## Datasets (8 repos)
| Repo | Rows | Downloads | Purpose |
|---|---|---|---|
| `sakthai-combined-v7` | 2,424 | **0** | Latest training data (en+th) |
| `sakthai-combined-v6` | 2,116 | 175 | Previous training data |
| `sakthai-irrelevance-supplement` | 60 | **0** | When NOT to call tools |
| `sakthai-bench-v2` | 500 | **0** | Official BFCL benchmark |
| `sakthai-bench-v1` | 235 | **0** | Early benchmark |
| `SimpleToolCalling` | — | 52 | Deprecated, gated |
| `food-penguin-v1` | 648 | 51 | Domain-specific (restaurant) |
| `sakthai-kaggle-notebooks` | — | 103 | Training notebooks |

## Spaces (3, all static HTML)
| Space | Topic | Hardware |
|---|---|---|
| `sakthai-tts` | TTS demo | None (static) |
| `sakthai-leaderboard` | Model leaderboard | None (static) |
| `sakthai-vision-demo` | Vision 7B demo | None (static) |

## ⚠️ Known issues
- **Benchmark scores inverted**: 0.5B (91.2%) > 1.5B (48.2%) — root cause: 0.5B uses all 7 LoRA targets + prompt-masked loss; 1.5B only uses 4 targets + chat data dilution
- **Plus LoRA not merged**: `sakthai-plus-1.5b-lora` has LoRA weights but hasn't been merged into base model or benchmarked yet
- **No interactive Spaces**: All 3 are static HTML, no GPU allocated

## Available commands
| Command | Usage |
|---|---|
| `/hf-explore` | Full inventory of Nanthasit repos |
| `/hf-plan` | Gap analysis + ecosystem roadmap |
| `/hf-publish` | Push models with model cards |
| `/hf-eval` | Run BFCL eval across models |
| `/hf-dataset` | Create/update dataset cards |
| `/hf-space` | Build interactive Gradio Space |
| `/hf-deploy` | Deploy to Inference Endpoint |
| `/hf-quantize` | Create GGUF quants |
| `/hf-agent` | Build smolagents agent |
| `/hf-bench` | Run BenchmarkQED eval |
| `/hf-cycle` | End-to-end development cycle |
| `/hf-loop` | 5-iteration learning improvement loop |
| `/hf-augment` | Generate augmented training data |
| `/hf-monitor` | Health check on downloads, likes, scores |
| `/hf-analyze-errors` | Deep-dive error analysis on eval failures |
| `/hf-cost` | Estimate and track training costs |
| `/hf-ollama` | Set up Ollama for local inference |
| `/hf-discover` | Platform optimization — widgets, Spaces, metadata, collections |
| `/hf-troubleshoot` | Diagnose errors in training, eval, HF Jobs |
| `/hf-profile` | Benchmark inference speed and latency |
| `/hf-migrate` | Version migration v1→v2, dataset versioning |
| `/hf-security` | Audit tokens, secrets, gated repos |
| `/hf-contribute` | Guide for community contributors |
| `/hf-auto` | Universal auto-command — detects and runs the right action |

### Autoskills (auto-trigger on keywords)
| Skill | Trigger keywords |
|---|---|
| `autoskill-hub` | upload, push to hub, model card, repo, downloads |
| `autoskill-data` | dataset, JSONL, data quality, v7, v8 |
| `autoskill-train` | train, fine-tune, HF Jobs, QLoRA, merge, GGUF |
| `autoskill-model` | model card, publish, version, deprecate, cross-link |
