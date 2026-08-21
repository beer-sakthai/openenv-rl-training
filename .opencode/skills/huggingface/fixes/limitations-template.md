## Limitations

- **Synthetic training data** — trained on synthetically generated tool-calling examples, which may not capture all real-world edge cases
- **Language bias** — primarily English; performance on other languages may be lower
- **No RLHF/DPO alignment** — fine-tuned with SFT only, no preference-based optimization
- **Self-reported benchmarks** — all evaluation scores are self-reported and not independently audited
- **Hardware assumptions** — benchmarked on specific GPU configurations; results may vary
- **Hallucination risk** — like all LLMs, may generate plausible but incorrect tool calls or arguments
- **Single-author project** — built by one person on free compute; not a commercial product
