import importlib.util
from pathlib import Path
import re

# Load train-sakthai-embedding.py via importlib
_script_path = Path(__file__).resolve().parent.parent / "train-sakthai-embedding.py"
_spec = importlib.util.spec_from_file_location("train_sakthai_embedding", _script_path)
tse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tse)


def test_tool_retrieval_triplets_integrity():
    triplets = tse.TOOL_RETRIEVAL_TRIPLETS
    assert len(triplets) >= 8, "Expected at least 8 bilingual triplets"

    thai_regex = re.compile(r"[\u0e00-\u0e7f]")
    has_thai = False
    has_english = False

    for anchor, pos, neg in triplets:
        assert isinstance(anchor, str) and anchor.strip()
        assert isinstance(pos, str) and pos.strip()
        assert isinstance(neg, str) and neg.strip()
        assert pos != neg, "Positive doc must differ from negative doc"

        if thai_regex.search(anchor):
            has_thai = True
        else:
            has_english = True

    assert has_thai, "Triplets must contain Thai language queries"
    assert has_english, "Triplets must contain English language queries"


def test_tool_catalog_docs():
    catalog = tse.TOOL_CATALOG_DOCS
    assert len(catalog) >= 8
    expected_tools = {
        "get_weather",
        "compute_vat",
        "lookup_order",
        "run_command",
        "write_file",
        "run_action",
        "get_stock_price",
        "send_email",
    }
    assert expected_tools.issubset(catalog.keys())
    for tool_name, doc in catalog.items():
        assert isinstance(doc, str) and len(doc) > 10
        assert tool_name in doc, f"Tool doc for {tool_name} should contain function name"


def test_eval_queries_consistency():
    queries = tse.EVAL_QUERIES
    assert len(queries) >= 6
    catalog = tse.TOOL_CATALOG_DOCS
    for query, gold_tool in queries:
        assert isinstance(query, str) and query.strip()
        assert gold_tool in catalog, f"Gold tool '{gold_tool}' not in catalog"


def test_build_triplet_dataset():
    dataset = tse.build_triplet_dataset()
    assert len(dataset) == len(tse.TOOL_RETRIEVAL_TRIPLETS)
    assert set(dataset.column_names) == {"anchor", "positive", "negative"}
    first = dataset[0]
    assert first["anchor"] == tse.TOOL_RETRIEVAL_TRIPLETS[0][0]
    assert first["positive"] == tse.TOOL_RETRIEVAL_TRIPLETS[0][1]
    assert first["negative"] == tse.TOOL_RETRIEVAL_TRIPLETS[0][2]
