#!/usr/bin/env python3
# /// script
# dependencies = ["huggingface_hub"]
# ///
"""
MCP-Bench adapter for sakthai-plus-1.5b.
Runs Plus through MCP-Bench evaluation framework.

Usage:
  HF_TOKEN=hf_... uv run python eval-mcp.py --help

Requires:
  - MCP-Bench cloned: git clone https://github.com/huggingface/mcp-bench.git
  - MCP servers installed: cd mcp_servers && bash install.sh
  - vLLM serving Plus: vllm serve Nanthasit/sakthai-plus-1.5b --port 8000
  - API keys configured in mcp_servers/api_key

Commands:
  HUGGINGFACE_BASE_URL=http://localhost:8000/v1 python run_benchmark.py \\
    --models huggingface --tasks-file tasks/mcpbench_tasks_single_runner_format.json
"""
import subprocess, sys, os

MODEL = "Nanthasit/sakthai-plus-1.5b"
MCP_BENCH_DIR = os.environ.get("MCP_BENCH_DIR", "./mcp-bench")

print("=" * 60)
print("MCP-Bench for sakthai-plus-1.5b")
print("=" * 60)
print(f"""
Setup steps:

1. Clone MCP-Bench:
   git clone https://github.com/huggingface/mcp-bench.git

2. Install MCP servers:
   cd mcp-bench/mcp_servers && bash install.sh

3. Start vLLM serving Plus:
   vllm serve {MODEL} --port 8000 \\
     --enable-auto-tool-choice --tool-call-parser hermes

4. Run MCP-Bench (single server tasks):
   cd mcp-bench
   HUGGINGFACE_BASE_URL=http://localhost:8000/v1 \\
   python run_benchmark.py --models huggingface \\
     --tasks-file tasks/mcpbench_tasks_single_runner_format.json

5. View results in benchmark_results_*.json

Or run on HF Jobs with H200:
   hf jobs uv run --flavor h200 --timeout 2h \\
     -e MCP_BENCH_DIR=/mcp-bench \\
     --secrets HF_TOKEN \\
     eval-mcp.py --hf-job

Expected score range: 0.3-0.5 (llama-3.1-8b = 0.428)
""")
