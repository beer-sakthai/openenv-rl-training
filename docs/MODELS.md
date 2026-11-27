# SakThai Model Portfolio & Ecosystem Reference

Comprehensive catalog of all models in the **SakThai AI ecosystem**, covering parameter scales, benchmark evaluations, empirical findings, and downstream runtime applications.

All public models, adapters, and datasets are hosted under the Hugging Face namespace [`Nanthasit`](https://huggingface.co/Nanthasit).

---

## 1. Master Model Portfolio Matrix (18 Verified Hub Models)

| Model Identifier | Base Architecture | Scale | Format | Primary Domain / Specialty | Key Capabilities & Targets | Hugging Face Link |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`sakthai-context-7b-tools`** | Qwen2.5-7B-Instruct | 7B | PEFT LoRA | **Flagship GRPO RL Base** | Solved 3/6 hermes tasks; non-zero reward variance (`~0.05`), gradient norms `0.25–0.49`. Targets: `k`, `q`, `o`, `v` | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-context-7b-tools) |
| **`sakthai-context-7b-merged`** | Qwen2.5-7B-Instruct | 7B | Merged Safetensors + GGUF | Full Flagship Model & Local Q8_0 | Full precision `bfloat16` weights + bundled `gguf/sakthai-context-7b-merged-Q8_0.gguf` | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-context-7b-merged) |
| **`sakthai-context-7b-tools-grpo`** | 7B + GRPO | 7B | Post-RL Checkpoint | Planned Aligned Checkpoint — not yet published | Designated target repo for RL training rollouts on OpenEnv environments | [🤗 Target repo](https://huggingface.co/Nanthasit/sakthai-context-7b-tools-grpo) |
| **`sakthai-context-7b-128k`** | Qwen2.5-7B-Instruct | 7B | Config / Adaptation | Extended Context Window | Extended **128,000-token context** RoPE frequency scaling via `yarn-128k-adapt.py` | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-context-7b-128k) |
| **`sakthai-context-1.5b-merged`** | Qwen2.5-1.5B-Instruct | 1.5B | Merged Safetensors + GGUF | Core 1.5B Production Model | **780+ downloads**; default base for Tier A (`env_simple_task.py`) and Tier B (`agent_tools/`) | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-context-1.5b-merged) |
| **`sakthai-context-1.5b-merged-v2`** | Qwen2.5-1.5B-Instruct | 1.5B | Merged Safetensors | Production v2 Checkpoint | Fine-tuned with rank-stabilized LoRA (rsLoRA) across all 7 linear targets; cross-model benchmarked | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-context-1.5b-merged-v2) |
| **`sakthai-context-1.5b-tools`** | Qwen2.5-1.5B-Instruct | 1.5B | PEFT LoRA | Lightweight Function Calling | Balanced tool calling on consumer GPUs / Colab T4. Targets: `q`, `k`, `o`, `v` | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-context-1.5b-tools) |
| **`sakthai-context-1.5b-tools-v2`** | Qwen2.5-1.5B-Instruct | 1.5B | PEFT LoRA | Upgraded Multi-Projection Adapter | LoRA adapter targeting **all 7 linear modules** (`q`, `k`, `v`, `o`, `gate`, `up`, `down`) | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-context-1.5b-tools-v2) |
| **`sakthai-context-0.5b-tools`** | Qwen2.5-0.5B-Instruct | 0.5B | Full Safetensors | High-Speed Single-Shot Router | **91.0% tool selection** on `sakthai-bench-v2`; zero rollout variance on RL (SFT-only) | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-context-0.5b-tools) |
| **`sakthai-context-0.5b-merged`** | Qwen2.5-0.5B-Instruct | 0.5B | Merged Safetensors + GGUF | Edge & Local Serving | **480+ downloads**; full precision weights + bundled `sakthai-0.5b-Q4_K_M.gguf` | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-context-0.5b-merged) |
| **`sakthai-context-0.5b-tools-sft-v2`** | Qwen2.5-0.5B-Instruct | 0.5B | PEFT LoRA | Upgraded v2 SFT Adapter | Fine-tuned adapter on top of `0.5b-tools` with chat template support | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-context-0.5b-tools-sft-v2) |
| **`sakthai-coder-browser`** | Qwen2.5-Coder-1.5B | 1.5B | Merged Safetensors | Browser Automation & Web Agent | DOM element reasoning and web interaction for BrowserGym MiniWoB++ | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-coder-browser) |
| **`sakthai-coder-browser-lora`** | Qwen2.5-Coder-1.5B | 1.5B | PEFT LoRA | Parameter-Efficient Web Adapter | LoRA adapter targeting all 7 linear projection layers on Qwen2.5-Coder-1.5B | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-coder-browser-lora) |
| **`sakthai-coder-1.5b`** | Qwen2.5-Coder-1.5B | 1.5B | GGUF Quantized | Local Code Sandbox Agent | `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` with verified benchmark evaluation records | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-coder-1.5b) |
| **`sakthai-plus-1.5b`** | Qwen2.5-1.5B | 1.5B | Merged Safetensors | Bilingual General Assistant | General conversational intelligence, reasoning, and Thai-English dialogue (MMLU / GSM8k) | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-plus-1.5b) |
| **`sakthai-vision-7b`** | Qwen2-VL-2B / LLaVA | 7B | GGUF Multimodal | Multimodal Visual Parsing | `llava-1.5-7b-hf-q4_k_m.gguf` + `mmproj-model-f16.gguf` for screenshot parsing & DOM localization | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-vision-7b) |
| **`sakthai-tts-model`** | Kokoro-82M | ~82M | GGUF Audio | Voice Synthesis | `kokoro-82m-q8_0.gguf` for low-latency Thai & English speech generation | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-tts-model) |
| **`sakthai-embedding-multilingual`** | Multilingual-MiniLM-L12 | ~384-dim | Sentence-Transformers | Vector Memory & Tool Filtering | Dense semantic text embeddings for RAG and pre-filtering candidate tool schemas | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-embedding-multilingual) |
| **`my-embedding-gemma`** | Gemma3-300M Text | Dense | Sentence-Transformers | High-Precision Semantic Routing | Dense semantic vector representations with Gemma 3 architecture | [🤗 Model Card](https://huggingface.co/Nanthasit/my-embedding-gemma) |

---

## 2. In-Depth Categorical Breakdown

### Category A: Tool-Calling & Reinforcement Learning Pipeline (Core Focus)

This core line is fine-tuned and aligned within `openenv-rl-training`:

1. **`sakthai-context-7b-tools`** / **`-merged`** (7B Parameters) — *Flagship RL Base*
   - **Empirical Role**: Solved 3/6 multi-turn coding tasks (`fix_failing_test`, `fix_broken_imports`, `add_verbose_flag`) in `hermes-tool-use-rl-env`.
   - **RL Status**: Demonstrates healthy, non-zero reward variance (`~0.05`) and stable gradient norms (`0.25–0.49`) during multi-turn rollouts. **The primary verified base model for GRPO**.
   - **Checkpoints**: Available as rank-16 PEFT LoRA adapter (`sakthai-context-7b-tools`), full `bfloat16` merged checkpoint (`sakthai-context-7b-merged`), and quantized local checkpoint (`gguf/sakthai-context-7b-merged-Q8_0.gguf`).
   - **RL Target**: Training runs push post-RL aligned weights to `Nanthasit/sakthai-context-7b-tools-grpo`.

2. **`sakthai-context-1.5b-merged`** / **`-merged-v2`** / **`-tools`** / **`-tools-v2`** (1.5B Parameters)
   - **Empirical Role**: Intermediate tool-calling models for single-step and multi-turn API queries on consumer hardware (Colab T4, 4–8 GB VRAM).
   - **Checkpoints**:
     - `sakthai-context-1.5b-merged`: Production merged checkpoint with **780+ downloads**; default base across Tier A and Tier B environments.
     - `sakthai-context-1.5b-merged-v2`: Production v2 model trained with rank-stabilized LoRA (rsLoRA) across all 7 linear targets.
     - `sakthai-context-1.5b-tools`: 4-target PEFT LoRA adapter (`q`, `k`, `o`, `v`).
     - `sakthai-context-1.5b-tools-v2`: Full 7-target PEFT LoRA adapter (`q`, `k`, `v`, `o`, `gate`, `up`, `down`).

3. **`sakthai-context-0.5b-tools`** / **`-merged`** / **`-tools-sft-v2`** (0.5B Parameters)
   - **Empirical Role**: Best single-shot tool selector (**91.0%** selection accuracy on `sakthai-bench-v2`), but incapable of sustaining multi-turn rollouts (0/6 on hermes).
   - **RL Status**: Zero-variance reward outputs during rollouts mean gradients remain 0; classified as **SFT-only**.
   - **Checkpoints**: Full safetensors (`0.5b-tools`), merged with local 4-bit GGUF (`0.5b-merged`), and upgraded v2 adapter (`0.5b-tools-sft-v2`).

---

### Category B: Browser Automation & Coding Agents

1. **`sakthai-coder-browser`** & **`sakthai-coder-browser-lora`** (1.5B Parameters)
   - **Specialty**: Fine-tuned on browser navigation trajectories, DOM element reasoning, and web action execution for BrowserGym MiniWoB++ tasks (`train.py --env browsergym`).
   - **Packaging**: Both full merged `bfloat16` checkpoint (`sakthai-coder-browser`) and 7-target LoRA adapter (`sakthai-coder-browser-lora`).

2. **`sakthai-coder-1.5b`** (1.5B Parameters)
   - **Specialty**: Quantized code completion, syntax repair, and test generation agent packaged in GGUF (`qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`) with verified benchmark test logs.

---

### Category C: General Intelligence & Reasoning ("Plus")

1. **`sakthai-plus-1.5b`** (1.5B Parameters)
   - **Specialty**: General knowledge, reasoning, and bilingual Thai-English dialogue.
   - **Evaluation**: Continuously benchmarked on reasoning suites (MMLU, GSM8k, HellaSwag) using [`sakthai-sft-training/eval-light.py`](../sakthai-sft-training/eval-light.py).

---

### Category D: Long-Context & Quantized Offline Deployments

1. **`sakthai-context-7b-128k`** (128k Context Window)
   - **Specialty**: RoPE frequency scaling configuration and adaptation script (`yarn-128k-adapt.py`) supporting up to **128,000 context tokens**.
   - **Primary Use Case**: Long-context document analysis, multi-chapter reference manuals, and whole-codebase repository ingestion.

2. **Quantized Local GGUF Weights**
   - Packaged directly within merged repositories for offline local execution via `llama.cpp` or Ollama:
     - `Nanthasit/sakthai-context-7b-merged/gguf/sakthai-context-7b-merged-Q8_0.gguf`
     - `Nanthasit/sakthai-context-0.5b-merged/sakthai-0.5b-Q4_K_M.gguf`
     - `Nanthasit/sakthai-coder-1.5b/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`

---

### Category E: Multimodal, Voice & Semantic Retrieval

1. **`sakthai-vision-7b`**
   - **Specialty**: Multimodal visual perception combining LLaVA / Qwen2-VL architecture with GGUF projection weights (`mmproj-model-f16.gguf`).
   - **Primary Use Case**: Visual UI parsing, screenshot analysis, and bounding-box localization in BrowserGym.

2. **`sakthai-tts-model`**
   - **Specialty**: Quantized Thai and English voice synthesis (`kokoro-82m-q8_0.gguf`).
   - **Primary Use Case**: Spoken voice output for interactive agent voice channels.

3. **`sakthai-embedding-multilingual`** & **`my-embedding-gemma`**
   - **Specialty**: Dense semantic vector embeddings for cross-lingual (Thai & English) similarity matching and tool routing.
   - **Training & Evaluation**: Fine-tuned with `SentenceTransformerTrainer` and `MultipleNegativesRankingLoss` on bilingual triplets via [`sakthai-sft-training/train-sakthai-embedding.py`](../sakthai-sft-training/train-sakthai-embedding.py).

---

## 3. Packaging Formats & Serving

All models are released in three standardized distribution formats:

- **LoRA Adapters (`*-tools`, `*-tools-v2`, `*-lora`)**: Lightweight rank-16 adapter weights (~50 MB–250 MB) allowing dynamic model loading and multi-tenant serving.
- **Full Precision Merged Checkpoints (`*-merged`, `*-merged-v2`)**: 16-bit `bfloat16` weights for production serving with vLLM, Hugging Face TGI, or TensorRT-LLM.
- **Quantized GGUF (`*.gguf`)**: Quantized integer representations (Q4_K_M, Q8_0) bundled inside model repositories for memory-constrained edge hardware.

---

## 4. Core Hugging Face Datasets

The repository models are trained and benchmarked against authoritative datasets hosted under `Nanthasit`:

| Dataset Identifier | Scale | Format | Description / Purpose |
| :--- | :--- | :--- | :--- |
| **`sakthai-bench-v3`** | 1,200 examples | JSONL / Parquet | Flagship tool-calling benchmark across `simple`, `parallel`, `irrelevance`, and `held-out` categories. |
| **`sakthai-bench-v2`** | 500 examples | JSONL / Parquet | Standard 500-example benchmark for cross-model comparative scoring (`eval-cross-model.py`). |
| **`sakthai-bench-v1`** | 200 examples | JSONL | Initial single-shot tool validation set. |
| **`sakthai-combined-v12`** | 7,437 examples | JSONL / Parquet | Latest expanded multi-turn conversational tool-calling SFT dataset. |
| **`sakthai-combined-v10`** | 5,800 examples | JSONL / Parquet | Stable baseline SFT dataset used in production runs. |
| **`sakthai-combined-v7`** | 3,200 examples | JSONL | Mid-stage data augmentation benchmark corpus. |
| **`sakthai-combined-v6`** | 2,400 examples | JSONL | Early SFT training corpus. |
| **`hermes-tool-use-rl-env`** | 6 tasks | JSONL / Env | 6-task agentic multi-turn coding environment used for RL evaluation. |
| **`sakthai-coder-browser`** | 247 examples | JSONL | Ground-truth browser navigation actions and code generation trajectories for BrowserGym. |
| **`sakthai-github-actions-v2`** | 850 examples | JSONL | Workflow generation, CI/CD step repair, and GitHub automation tasks. |
| **`sakthai-irrelevance-supplement`** | 300 examples | JSONL | Negative sampling dataset for detecting irrelevant / no-tool prompts. |
| **`sakthai-kaggle-notebooks`** | Multi-file | JSON / Notebooks | Canonical Kaggle notebook artifacts and runner checkpoints. |
| **`sakthai-openenv-training`** | Runtime Pins | JSON / YAML | Canonical runtime dependencies and model pinning dataset. |

---

## 5. Downstream Runtime Architecture

The models in this repository are orchestrated downstream by [`beer-sakthai/Sak-Family-Agent`](https://github.com/beer-sakthai/Sak-Family-Agent):

```text
                                  ┌── sakthai-embedding-multilingual (Vector Memory)
                                  ├── sakthai-vision-7b (Screenshot / UI Parsing)
[ Sak-Family-Agent ] ──orchestrates─┼── sakthai-context-7b-tools-grpo (Action / Tool Loop, planned)
                                  ├── sakthai-coder-1.5b (Code Execution Sandbox)
                                  └── sakthai-tts-model (Spoken Audio Output)
```

For empirical training transcripts, consult [`sakthai-agentic-eval-train/FINDINGS.md`](../sakthai-agentic-eval-train/FINDINGS.md).
