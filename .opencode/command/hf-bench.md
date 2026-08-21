---
description: Set up and run Microsoft BenchmarkQED (AutoQ + AutoE + AutoD) for evaluating SakThai models — synthetic query generation, LLM-as-a-judge scoring, and dataset curation.
agent: general
---

Help the user set up BenchmarkQED to evaluate their SakThai tool-calling models. This replaces/supplements the custom BFCL eval scripts with Microsoft's automated benchmarking framework.

1. Read the benchmark-qed skill for reference.

2. **Scaffold a benchmark project**:
   ```bash
   mkdir -p ./sakthai-bench-qed/input
   cd ./sakthai-bench-qed
   ```

3. **Download the SakThai bench dataset** (or use your own):
   - Point to `Nanthasit/sakthai-bench-v2` as the evaluation corpus
   - Or use raw documents from a domain-specific dataset

4. **Initialize AutoQ config**:
   ```bash
   benchmark-qed config init autoq .
   ```
   Edit `.env` to add OpenAI or Azure API key for the LLM judge.
   Edit `settings.yaml` to configure query distribution and count.

5. **Generate synthetic queries**:
   ```bash
   benchmark-qed autoq settings.yaml output
   ```
   This creates queries with assertions (testable facts).

6. **Run SakThai model inference** against the generated queries:
   - Write a script that loads `Nanthasit/sakthai-context-1.5b-merged`
   - Generates answers for each query
   - Saves to `./input/answers_sakthai.json`

7. **Initialize and run AutoE**:
   ```bash
   benchmark-qed config init autoe_pairwise .
   benchmark-qed autoe pairwise-scores settings.yaml output
   ```

8. **For before/after comparison** (the current eval pattern):
   - Generate answers from v1 (`sakthai-context-1.5b-merged`) and v2 (future)
   - Run AutoE pairwise to get win rates

Return a complete step-by-step guide with the specific file contents needed.
