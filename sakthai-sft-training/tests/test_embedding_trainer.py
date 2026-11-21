"""Unit tests for train-sakthai-embedding.py (multilingual embedding & tool-retrieval)."""

import os
import sys

SFT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

# Import module with hyphenated name using __import__
embedding_trainer = __import__("train-sakthai-embedding")


def test_embedding_triplets_and_catalog_integrity():
    """Verify triplets have valid anchors, positive, and negative docstrings."""
    catalog = embedding_trainer.TOOL_CATALOG_DOCS
    triplets = embedding_trainer.TOOL_RETRIEVAL_TRIPLETS

    assert len(catalog) >= 8
    assert len(triplets) >= 8

    for anchor, pos, neg in triplets:
        assert isinstance(anchor, str) and len(anchor.strip()) > 0
        assert isinstance(pos, str) and len(pos.strip()) > 0
        assert isinstance(neg, str) and len(neg.strip()) > 0
        assert pos != neg, f"Positive and negative must differ: {pos!r}"


def test_embedding_eval_queries_integrity():
    """Verify eval queries map to valid tools in the catalog and include bilingual entries."""
    catalog = embedding_trainer.TOOL_CATALOG_DOCS
    queries = embedding_trainer.EVAL_QUERIES

    assert len(queries) >= 6
    thai_count = 0
    en_count = 0

    for query, tool_name in queries:
        assert tool_name in catalog, f"Eval tool {tool_name!r} not found in TOOL_CATALOG_DOCS"
        # Check Thai characters vs English
        if any('\u0e00' <= char <= '\u0e7f' for char in query):
            thai_count += 1
        else:
            en_count += 1

    assert thai_count >= 2, "Expected at least 2 Thai eval queries"
    assert en_count >= 2, "Expected at least 2 English eval queries"



def test_build_triplet_dataset():
    """Verify build_triplet_dataset constructs a Hugging Face Dataset with expected columns."""
    ds = embedding_trainer.build_triplet_dataset()
    assert len(ds) == len(embedding_trainer.TOOL_RETRIEVAL_TRIPLETS)
    assert set(ds.column_names) == {"anchor", "positive", "negative"}

    first_row = ds[0]
    assert first_row["anchor"]
    assert first_row["positive"]
    assert first_row["negative"]



def test_parse_args_defaults_and_eval_only(monkeypatch):
    """Verify parse_args handles default models and --eval-only."""
    monkeypatch.setattr(sys, "argv", ["train-sakthai-embedding.py", "--eval-only"])
    args = embedding_trainer.parse_args()
    assert args.eval_only is True
    assert args.base_model == embedding_trainer.DEFAULT_BASE_MODEL
