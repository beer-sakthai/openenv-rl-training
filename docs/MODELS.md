# SakThai Model Portfolio & Ecosystem Reference

Comprehensive catalog of all models in the **SakThai AI ecosystem**, covering parameter scales, benchmark evaluations, empirical findings, and downstream runtime applications.

All public models, adapters, and datasets are hosted under the Hugging Face namespace [`Nanthasit`](https://huggingface.co/Nanthasit).

---

## 1. Master Model Portfolio Matrix

| Model Identifier | Base Architecture | Scale | Primary Domain / Specialty | Deployment Target | Hugging Face Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`sakthai-context-0.5b-tools`** | Qwen2.5-0.5B-Instruct | 0.5B | High-speed single-shot tool routing (**91.0%** selection) | Edge / CPU / Mobile | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-context-0.5b-tools) |
| **`sakthai-context-1.5b-tools`** | Qwen2.5-1.5B-Instruct | 1.5B | Balanced lightweight tool calling & JSON output | Colab T4 / Consumer GPUs | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-context-1.5b-tools) |
| **`sakthai-context-7b-tools`** | Qwen2.5-7B-Instruct | 7B | Flagship base model; multi-turn agentic reasoning & coding | Heavy GPU / vLLM / GRPO Base | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-context-7b-tools) |
| **`sakthai-context-7b-tools-grpo`**| 7B + GRPO | 7B | Production RL-aligned checkpoint (BrowserGym & terminal tasks) | Autonomous Agent Runtime | [🤗 Hub Target](https://huggingface.co/Nanthasit/sakthai-context-7b-tools-grpo) |
| **`sakthai-context-7b-128k`** | Qwen2.5-7B | 7B | Extended **128,000-token context window** for document analysis | Long-Context RAG / Search | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-context-7b-128k) |
| **`sakthai-context-7b-gguf`** | Qwen2.5-7B (Quantized) | 7B | 4-bit / 8-bit quantized weights for offline local inference | `llama.cpp` / Ollama / Mac | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-context-7b-gguf) |
| **`sakthai-coder-1.5b`** | Qwen2.5-Coder-1.5B | 1.5B | Specialized code completion, syntax fixing, and test generation | Coding Agent Sandbox | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-coder-1.5b) |
| **`sakthai-plus-1.5b`** | Qwen2.5-1.5B | 1.5B | General Thai-English conversational assistant (MMLU / GSM8k) | General Conversational Chat | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-plus-1.5b) |
| **`sakthai-plus-1.5b-coder`** | Qwen2.5-1.5B-Coder | 1.5B | Coding assistant variant tuned on reasoning data | Developer Tools & CLI | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-plus-1.5b-coder) |
| **`sakthai-plus-1.5b-lora`** | Qwen2.5-1.5B | 1.5B | Parameter-efficient adapter for general conversational tasks | Dynamic Adapter Serving | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-plus-1.5b-lora) |
| **`sakthai-vision-7b`** | Qwen2-VL / Vision Base | 7B | Multimodal vision-language understanding, UI parsing & OCR | Browser Automation / Vision | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-vision-7b) |
| **`sakthai-tts-model`** | Voice Synthesis Base | ~0.3B | Thai & English voice synthesis and spoken responses | Voice UI & Agent Audio | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-tts-model) |
| **`sakthai-embedding-multilingual`**| Multilingual Dense | ~0.3B | Semantic text embedding for vector database retrieval | RAG & Vector Memory | [🤗 Model Card](https://huggingface.co/Nanthasit/sakthai-embedding-multilingual) |

---

## 2. In-Depth Categorical Breakdown

### Category A: Tool-Calling & RL Pipeline (This Repository)

This core line is fine-tuned and aligned within `openenv-rl-training`:

1. **`sakthai-context-0.5b-tools`** / **`-merged`** (0.5B Parameters)
   - **Empirical Role**: Best single-shot tool selector (**91.0%** selection on `sakthai-bench-v2`), but incapable of sustaining multi-turn rollouts (0/6 on hermes).
   - **RL Status**: Zero-variance reward outputs during rollouts mean gradients remain 0; marked as **SFT-only**.
   - **Best For**: Rapid query classification and high-speed tool routing at < 1 GB VRAM.

2. **`sakthai-context-1.5b-tools`** / **`-merged`** (1.5B Parameters)
   - **Empirical Role**: Intermediate tool-calling assistant for single-step API queries and lightweight function execution.
   - **Best For**: Cost-effective deployment on consumer GPUs (4–8 GB VRAM) and Colab T4.

3. **`sakthai-context-7b-tools`** / **`-merged`** (7B Parameters) — *Flagship Base*
   - **Empirical Role**: Solved 3/6 multi-turn coding tasks (`fix_failing_test`, `fix_broken_imports`, `add_verbose_flag`) in `hermes-tool-use-rl-env`.
   - **RL Status**: Demonstrates strong, non-zero reward variance and stable gradient norms (0.25–0.49) during rollouts. **The primary base model for GRPO**.
   - **Best For**: Complex multi-step reasoning, nested JSON parameter extraction, and autonomous problem solving.

4. **`sakthai-context-7b-tools-grpo`** (Post-RL Target Checkpoint)
   - **Empirical Role**: Final policy aligned through `GRPOTrainer` on interactive [Hugging Face OpenEnv](https://github.com/huggingface/OpenEnv) environments (BrowserGym MiniWoB++ web navigation, coding sandboxes).
   - **Best For**: End-to-end agentic workflows requiring live environment feedback and self-correction.

---

### Category B: Long-Context & Quantized Offline Deployments

1. **`sakthai-context-7b-128k`** (128k Context Window)
   - **Specialty**: Trained with extended RoPE scaling to support up to **128,000 context tokens**.
   - **Primary Use Case**: Ingesting entire codebases, multi-chapter documentation, or extensive legal/technical documents without chunking degradation.
   - **Community Traction**: Over 500+ organic downloads on the Hugging Face Hub.

2. **`sakthai-context-7b-gguf`** (Quantized Local Weights)
   - **Specialty**: 4-bit (`Q4_K_M`) and 8-bit quantized weights packaged for CPU/Apple Silicon execution.
   - **Primary Use Case**: Running private, air-gapped, or local agent instances via `llama.cpp` or Ollama without dedicated GPU servers.

---

### Category C: Specialized Coding & General Intelligence ("Plus")

1. **`sakthai-coder-1.5b`** & **`sakthai-plus-1.5b-coder`**
   - **Specialty**: Fine-tuned on coding benchmarks, syntax repair, and automated test suite creation.
   - **Primary Use Case**: Dedicated code-generation subagent running inside sandbox containers.

2. **`sakthai-plus-1.5b`** & **`sakthai-plus-1.5b-lora`**
   - **Specialty**: General knowledge, reasoning, and conversational Thai-English communication.
   - **Evaluation**: Continuously benchmarked on standard reasoning suites (MMLU, GSM8k, HellaSwag) using [`sakthai-sft-training/eval-light.py`](../sakthai-sft-training/eval-light.py).

---

### Category D: Multimodal, Voice & Semantic Retrieval

1. **`sakthai-vision-7b`**
   - **Specialty**: Multimodal visual understanding.
   - **Primary Use Case**: Web page screenshot analysis, DOM element visual localization in BrowserGym, and diagram comprehension.

2. **`sakthai-tts-model`**
   - **Specialty**: Thai and English text-to-speech generation.
   - **Primary Use Case**: Conversational voice agent responses for interactive voice assistant channels.

3. **`sakthai-embedding-multilingual`**
   - **Specialty**: High-dimensional dense vector embeddings optimized for cross-lingual (Thai & English) semantic retrieval.
   - **Primary Use Case**: Powering the long-term vector memory store in the agent runtime.

---

## 3. Packaging Formats & Serving

All models are released in three standardized distribution formats:

- **LoRA Adapters (`*-tools`, `*-lora`)**: Lightweight rank-16 adapter weights (~50 MB–250 MB) allowing dynamic model loading and multi-tenant serving.
- **Full Precision Merged Checkpoints (`*-merged`)**: 16-bit `bfloat16` weights for production serving with vLLM, Hugging Face TGI, or TensorRT-LLM.
- **Quantized GGUF (`*-gguf`)**: Quantized integer representations for memory-constrained edge hardware.

---

## 4. Downstream Runtime Architecture

The models in this repository are orchestrated downstream by [`beer-sakthai/Sak-Family-Agent`](https://github.com/beer-sakthai/Sak-Family-Agent):

```text
                                  ┌── sakthai-embedding-multilingual (Vector Memory)
                                  ├── sakthai-vision-7b (Screenshot / UI Parsing)
[ Sak-Family-Agent ] ──orchestrates─┼── sakthai-context-7b-tools-grpo (Action / Tool Loop)
                                  ├── sakthai-coder-1.5b (Code Execution Sandbox)
                                  └── sakthai-tts-model (Spoken Audio Output)
```

For empirical training transcripts, consult [`sakthai-agentic-eval-train/FINDINGS.md`](../sakthai-agentic-eval-train/FINDINGS.md).
