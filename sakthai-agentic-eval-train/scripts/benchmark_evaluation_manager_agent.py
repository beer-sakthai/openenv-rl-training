"""Score the evaluation manager agent against repository-specific scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGENT = REPO_ROOT / ".claude/agents/openenv-rl-training-evaluation-manager.md"

SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "name": "single-shot benchmark",
        "request": "Run the PEFT-aware sakthai-bench-v2 evaluator.",
        "terms": ("sakthai-bench-v2", "eval_bench_peft.py", "PEFT"),
    },
    {
        "name": "agentic evaluation",
        "request": "Diagnose a multi-turn Hermes tool-use score and inspect transcripts.",
        "terms": (
            "hermes-tool-use-rl-env",
            "eval_hermes_env.py",
            "SAK_RENDER=native",
            "SAK_DUMP=1",
        ),
    },
    {
        "name": "pipeline comparison",
        "request": "Compare checkpoints and update the durable findings record.",
        "terms": (
            "model, adapter, dataset, renderer",
            "FINDINGS.md",
            "provenance",
            "metrics",
        ),
    },
    {
        "name": "remote evaluation operation",
        "request": "Prepare a reproducible Hugging Face Jobs evaluation.",
        "terms": (
            "lighteval.yml",
            "mcp-bench.yml",
            "paid",
            "402 Payment Required",
        ),
    },
)

REQUIRED_GUIDANCE: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "execution boundary",
        (
            "Never report an unrun GPU or paid Hugging Face Jobs operation",
            "Do not dispatch HF Jobs",
            "tokens out of commands",
        ),
    ),
    (
        "reproducibility",
        (
            "environment revision",
            "dataset and revision",
            "job ID when remote",
            "output destinations",
        ),
    ),
    (
        "delivery workflow",
        (
            "feature branch",
            "all applicable checks are green",
            "explicitly confirmed",
            "Output format",
        ),
    ),
)


def parse_agent(path: Path) -> tuple[str, str]:
    """Return frontmatter and prompt from an agent markdown file."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError("agent must start with YAML frontmatter")
    parts = text.split("---", 2)
    if len(parts) != 3:
        raise ValueError("agent frontmatter is not closed")
    return parts[1], parts[2]


def contains(text: str, term: str) -> bool:
    """Match a benchmark term case-insensitively."""
    return term.casefold() in text.casefold()


def score_agent(path: Path) -> dict[str, Any]:
    """Return scenario and guidance scores for an evaluation manager agent."""
    frontmatter, prompt = parse_agent(path)
    full_text = f"{frontmatter}\n{prompt}"
    scenario_results = []
    for scenario in SCENARIOS:
        matched = [term for term in scenario["terms"] if contains(full_text, term)]
        missing = [term for term in scenario["terms"] if term not in matched]
        scenario_results.append(
            {
                "name": scenario["name"],
                "request": scenario["request"],
                "matched": matched,
                "missing": missing,
                "score": len(matched) / len(scenario["terms"]),
            }
        )

    guidance_results = []
    for category, terms in REQUIRED_GUIDANCE:
        matched = [term for term in terms if contains(full_text, term)]
        missing = [term for term in terms if term not in matched]
        guidance_results.append(
            {
                "category": category,
                "matched": matched,
                "missing": missing,
                "score": len(matched) / len(terms),
            }
        )

    all_results = scenario_results + guidance_results
    overall_score = sum(result["score"] for result in all_results) / len(all_results)
    return {
        "agent": str(path),
        "scenarios": scenario_results,
        "guidance": guidance_results,
        "overall_score": overall_score,
        "passed": overall_score >= 0.9
        and all(not result["missing"] for result in scenario_results),
    }


def render_text(report: dict[str, Any]) -> str:
    """Render a compact human-readable benchmark report."""
    lines = [f"Agent benchmark: {report['agent']}", ""]
    for result in report["scenarios"]:
        lines.append(f"{result['name']}: {result['score']:.0%}")
        if result["missing"]:
            lines.append(f"  missing: {', '.join(result['missing'])}")
    lines.append("")
    for result in report["guidance"]:
        lines.append(f"{result['category']}: {result['score']:.0%}")
        if result["missing"]:
            lines.append(f"  missing: {', '.join(result['missing'])}")
    lines.append("")
    lines.append(f"Overall score: {report['overall_score']:.0%}")
    lines.append(f"Result: {'PASS' if report['passed'] else 'FAIL'}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=DEFAULT_AGENT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    report = score_agent(args.agent)
    output = json.dumps(report, indent=2) if args.as_json else render_text(report)
    print(output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
