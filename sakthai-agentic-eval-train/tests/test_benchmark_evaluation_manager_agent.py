from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_evaluation_manager_agent import (
    DEFAULT_AGENT,
    parse_agent,
    render_text,
    score_agent,
)


def test_benchmark_scores_the_repository_agent():
    report = score_agent(DEFAULT_AGENT)

    assert report["passed"]
    assert report["overall_score"] == 1.0
    assert all(not result["missing"] for result in report["scenarios"])


def test_benchmark_report_has_machine_and_human_facing_fields():
    report = score_agent(DEFAULT_AGENT)
    rendered = render_text(report)

    assert "Overall score: 100%" in rendered
    assert "Result: PASS" in rendered
    assert len(report["guidance"]) == 3


def test_benchmark_detects_missing_agent_guidance(tmp_path: Path):
    frontmatter, prompt = parse_agent(DEFAULT_AGENT)
    reduced_agent = tmp_path / "agent.md"
    reduced_agent.write_text(f"---{frontmatter}---{prompt.replace('SAK_DUMP=1', '')}", encoding="utf-8")

    report = score_agent(reduced_agent)

    assert not report["passed"]
    agentic = next(result for result in report["scenarios"] if result["name"] == "agentic evaluation")
    assert "SAK_DUMP=1" in agentic["missing"]
