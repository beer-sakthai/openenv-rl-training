---
name: benchmark-qed
description: Use when the user mentions benchmark-qed, AutoQ, AutoE, AutoD, LLM-as-a-judge, RAG evaluation, synthetic queries, assertion-based scoring, or wants to replace/reinforce their BFCL-style eval with Microsoft's automated benchmarking framework. Phase: EVAL.
---

# BenchmarkQED — Automated RAG/Tool-Calling Evaluation

**Website**: https://microsoft.github.io/benchmark-qed/
**Source**: https://github.com/microsoft/benchmark-qed
**PyPI**: `benchmark-qed`

## Overview

Microsoft's framework for automated benchmarking with three components:

```
AutoQ (query synthesis) ──→ AutoE (LLM judge evaluation) ←── AutoD (dataset curation)
```

### AutoQ — Generate synthetic queries
Creates query classes along two dimensions:
- **Scope**: Local (specific details) vs Global (themes/trends)
- **Source**: Data-driven (from corpus) vs Activity-driven (potential use cases)
- Also generates **assertions** — factual "unit tests" for answers
- Supports **data-linked** queries (multi-hop: bridge, comparison, intersection, temporal)

### AutoE — LLM-as-a-Judge evaluation
Three modes:
1. **Pairwise comparison** — compare RAG methods head-to-head on relevance, comprehensiveness, diversity, empowerment → win rates
2. **Reference-based scoring** — score answers against ground truth on correctness, completeness
3. **Assertion-based scoring** — binary pass/fail against generated assertions

### AutoD — Data utilities
- **Sampling**: subset datasets to target breadth (clusters) × depth (samples/cluster)
- **Summarization**: map-reduce topic summaries for prompts

## Why this matters for SakThai

Your current eval (`eval-sakthai-1.5b.py`) uses a simple regex scorer:
- Parses `<tool_call>` blocks
- Checks function name matches
- Basic pass/fail per category

BenchmarkQED can **supersede** this with:
| Limitation | BenchmarkQED fix |
|---|---|
| Simple name matching | LLM judges semantic correctness of arguments |
| No gradation | Pairwise win rates + multi-metric scoring |
| Manual test set | AutoQ generates diverse synthetic queries |
| No assertion testing | Auto-generated assertions as unit tests |
| One-off scripts | Standardized CLI pipeline |

## Mapping to tool-calling eval

| BFCL concept | BenchmarkQED equivalent |
|---|---|
| Test queries | AutoQ data-driven local queries |
| Gold tool calls | Assertions (the model MUST call get_weather) |
| Categories (simple/parallel/irrelevance) | Query classes with different complexity |
| Before/after comparison | AutoE pairwise comparison |
| Held-out tools | Custom assertion sets |

## CLI pipeline

```bash
# 1. Init config
benchmark-qed config init autoq .
benchmark-qed config init autoe_pairwise .

# 2. Generate synthetic queries from your dataset
benchmark-qed autoq settings.yaml output

# 3. Run model inference (your script generates answers for each query)

# 4. Evaluate with LLM judge
benchmark-qed autoe pairwise-scores settings.yaml output
```

## Key config files
- `.env` — API keys (OpenAI or Azure)
- `settings.yaml` — pipeline parameters (query count, distribution, LLM model)

## Important caveats
- Requires LLM API access (OpenAI/Azure) for the judge — adds cost
- A/A tests recommended to validate judge reliability
- Assertions may contain LLM inaccuracies — review before production use
- Currently designed for RAG — tool-calling eval needs custom answer format
