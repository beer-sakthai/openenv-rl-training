#!/usr/bin/env python3
# /// script
# dependencies = ["lighteval", "torch", "transformers", "datasets", "accelerate"]
# ///
"""Lighteval benchmark for sakthai-plus-1.5b.
Compares against Qwen2.5-1.5B-Instruct on standard benchmarks.

Usage:
  uv run python eval-light.py                          # Plus only, default tasks
  uv run python eval-light.py --tasks mmlu,gsm8k       # Specific tasks
  uv run python eval-light.py --compare                # Plus vs Qwen baseline
"""
import subprocess, sys, os

MODEL = "Nanthasit/sakthai-plus-1.5b"
BASELINE = "Qwen/Qwen2.5-1.5B-Instruct"
TASKS = sys.argv[2] if len(sys.argv) > 2 else "mmlu:0|gsm8k:0|hellaswag:0|winogrande:0"
COMPARE = "--compare" in sys.argv

models = [MODEL]
if COMPARE:
    models.append(BASELINE)

for model in models:
    name = model.split("/")[-1]
    print(f"\n=== Evaluating {name} ===")
    subprocess.run([
        sys.executable, "-m", "lighteval", "accelerate",
        f"--model_args=pretrained={model},dtype=bfloat16",
        f"--tasks={TASKS}",
        f"--output_dir=./results/{name}",
        "--save_details"
    ])

# Push results to HF Hub model card
print("\nPushing results to HF Hub...")
subprocess.run([
    sys.executable, "-m", "lighteval", "push_results",
    f"--output_dir=./results",
    f"--push_results_to_hub={MODEL}",
    "--push_metrics_to_hub"
])
print(f"Results: https://huggingface.co/{MODEL}")
print(f"Raw data: ./results/")
