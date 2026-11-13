# SakThai Model Family Reference

Comprehensive guide to all models in the SakThai family, their empirical performance, architectural trade-offs, and deployment targets.

All public model weights and adapters are published under the Hugging Face namespace [`Nanthasit`](https://huggingface.co/Nanthasit).

---

## 1. Capabilities & Benchmark Matrix

The models were evaluated against single-shot tool benchmarks (`sakthai-bench-v2`, 500 rows) and multi-step agentic coding environments (`hermes-tool-use-rl-env`, 6 multi-turn tasks):

| Model Tier | Base Model | Single-Shot Selection | Single-Shot Arguments | Multi-Turn Agentic (Hermes) | RL / GRPO Viability | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`0.5B Tools`** | Qwen2.5-0.5B-Instruct | **91.0%** (Highest) | 45.7% | 0/6 (Fails) | ❌ **No** (Zero gradient variance) | High-speed tool routing, intent classification, edge CPU/mobile |
| **`1.5B Tools`** | Qwen2.5-1.5B-Instruct | 55.8% | 11.0% | 0/6 (Fails) | ⚠️ **Not targeted** | Balanced low-resource GPUs (Colab T4, 4–8 GB VRAM) |
| **`7B Tools`** | Qwen2.5-7B-Instruct | 56.4% | 12.3% | **3/6** (Passed) | ✅ **Yes** (Real gradient signal: 0.25–0.49) | Flagship base model, complex reasoning, multi-turn tool calling |
| **`7B Tools GRPO`** | 7B + GRPO | Evaluated in-loop | Dynamic | Multi-turn web & coding | 🎯 **Final RL output** | End-to-end autonomous agents (BrowserGym web automation, coding) |

*Full empirical evaluation details and transcripts are recorded in [`sakthai-agentic-eval-train/FINDINGS.md`](../sakthai-agentic-eval-train/FINDINGS.md).*

---

## 2. In-Depth Model Breakdown

### 🔹 `sakthai-context-0.5b-tools` (0.5B Parameters)
- **Hugging Face**: [`Nanthasit/sakthai-context-0.5b-tools`](https://huggingface.co/Nanthasit/sakthai-context-0.5b-tools)
- **Role**: Ultra-lightweight SFT tool-routing adapter.
- **Key Benefits**:
  - **Highest Single-Shot Selection (91.0%)**: Best model in the family for simply picking the correct function from a candidate tool schema.
  - **Minimal Compute**: < 1 GB VRAM requirement; runs on CPU, mobile, or edge gateways with negligible latency.
- **Limitations**:
  - **Zero Agentic Success**: Cannot maintain state over multi-step tasks (0/6 on hermes).
  - **GRPO Dead End**: Because rollout success is 0%, reward variance is 0 throughout training, producing zero gradient updates. Marked **SFT-only**.

### 🔹 `sakthai-context-1.5b-tools` (1.5B Parameters)
- **Hugging Face**: [`Nanthasit/sakthai-context-1.5b-tools`](https://huggingface.co/Nanthasit/sakthai-context-1.5b-tools)
- **Role**: Intermediate model for single-GPU and consumer hardware.
- **Key Benefits**:
  - Operates comfortably within 4–8 GB VRAM (e.g., free Google Colab T4 GPUs).
  - Better language coherence and formatting robustness than the 0.5B model.
- **Target Use Cases**:
  - Single-step tool calls, structured JSON generation, and lightweight automation where 7B is too resource-heavy.

### 🔹 `sakthai-context-7b-tools` (7B Parameters) — Flagship Base Adapter
- **Hugging Face**: [`Nanthasit/sakthai-context-7b-tools`](https://huggingface.co/Nanthasit/sakthai-context-7b-tools)
- **Role**: The core foundation model of the SakThai family and the **only viable target for RL/GRPO**.
- **Key Benefits**:
  - **Multi-Turn Problem Solving**: Solved 3 of 6 complex multi-step coding tasks (`fix_failing_test`, `fix_broken_imports`, `add_verbose_flag`) in `hermes-tool-use-rl-env`.
  - **Proven RL Foothold**: Produces nonzero rollout rewards (~0.05) and stable gradient norms (0.25–0.49) during GRPO rollouts, providing the necessary signal for policy improvement.
  - Superior argument extraction, nested JSON formatting, and contextual comprehension.

### 🔹 `sakthai-context-7b-tools-grpo` (Post-RL Checkpoint)
- **Role**: The production reinforcement-learned model produced by running `GRPOTrainer` across interactive OpenEnv environments.
- **Key Benefits over SFT**:
  - **Feedback-Driven Correction**: Unlike SFT (which only mimics static demonstrations), GRPO optimizes policies against environment feedback (compiler logs, browser DOM state, task success signals).
  - **Interactive Automation**: Aligned for BrowserGym MiniWoB++ web workflows and terminal coding execution.

---

## 3. Artifact Formats & Serving

Each model tier is published in three packaging formats:

1. **LoRA Adapters (`*-tools`)**:
   - Small weight deltas (~50 MB for 0.5B to ~250 MB for 7B).
   - Easily mounted onto the base `Qwen/Qwen2.5-*-Instruct` model at runtime.
2. **Merged Models (`*-merged`)**:
   - Standalone 16-bit (`bfloat16`) checkpoints.
   - Ready for high-throughput serving with vLLM, TGI, or Hugging Face Inference Endpoints.
3. **Quantized GGUF (`*-gguf`)**:
   - 4-bit (`Q4_K_M`) and 8-bit quantized weights.
   - Optimized for local, offline execution using `llama.cpp` or Ollama on laptops without dedicated GPU clusters.

---

## 4. Downstream Deployment

Trained models are deployed into [`beer-sakthai/Sak-Family-Agent`](https://github.com/beer-sakthai/Sak-Family-Agent), which wraps them with:
- Persistent SQLite memory
- Model Context Protocol (MCP) tool servers
- Thai conversational personas
