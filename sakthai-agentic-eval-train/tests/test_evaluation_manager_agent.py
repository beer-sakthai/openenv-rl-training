from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_PATH = REPO_ROOT / ".claude" / "agents" / "openenv-rl-training-evaluation-manager.md"


def read_agent() -> tuple[str, str]:
    text = AGENT_PATH.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    assert len(parts) == 3
    return parts[1], parts[2]


def frontmatter_value(frontmatter: str, key: str) -> str:
    for line in frontmatter.splitlines():
        if line.startswith(f"{key}:"):
            return line.split(":", 1)[1].strip().strip('"')
    raise AssertionError(f"Missing frontmatter field: {key}")


def test_evaluation_manager_agent_has_valid_identity():
    frontmatter, prompt = read_agent()

    assert frontmatter_value(frontmatter, "name") == "openenv-rl-training-evaluation-manager"
    assert frontmatter_value(frontmatter, "model") == "inherit"
    assert frontmatter_value(frontmatter, "color") == "cyan"
    assert "Use this agent when managing evaluation pipelines" in frontmatter
    assert frontmatter.count("<example>") == 4
    assert len(prompt.strip()) >= 500


def test_evaluation_manager_agent_covers_all_evaluation_surfaces():
    _, prompt = read_agent()

    required_references = (
        "sakthai-bench-v2",
        "eval_bench_peft.py",
        "eval_hermes_env.py",
        "hermes-tool-use-rl-env",
        "sakthai-sft-training/eval-light.py",
        "sakthai-sft-training/eval-mcp.py",
        "FINDINGS.md",
        "lighteval.yml",
        "mcp-bench.yml",
    )
    for reference in required_references:
        assert reference in prompt, f"Missing evaluation reference: {reference}"


def test_evaluation_manager_agent_requires_provenance_and_honest_results():
    _, prompt = read_agent()

    required_guidance = (
        "model, adapter, dataset, renderer",
        "environment revision",
        "SAK_RENDER=native",
        "SAK_DUMP=1",
        "proof-of-signal",
        "402 Payment Required",
        "Never report an unrun GPU or paid Hugging Face Jobs operation",
        "tokens out of commands",
        "Merge only after all applicable checks are green",
    )
    for guidance in required_guidance:
        assert guidance in prompt, f"Missing safety or provenance guidance: {guidance}"


def test_evaluation_manager_agent_has_operational_sections():
    _, prompt = read_agent()

    for heading in (
        "## When to invoke",
        "## Core responsibilities",
        "## Repository evaluation map",
        "## Execution and safety rules",
        "## Analysis process",
        "## Pull request workflow",
        "## Output format",
    ):
        assert heading in prompt
