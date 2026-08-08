"""Tests for app.funds.retrieve.

Uses a small synthetic chunk list + in-memory FAISS index (via
monkeypatching load_index/_get_model) rather than the real built index, so
these tests don't depend on `python -m app.funds.index` having been run.
"""

import faiss
import numpy as np
import pytest

from app.funds import retrieve as retrieve_module
from app.funds.fund_catalog import TIER_TO_FUNDS, get_pitchable_fund
from app.funds.ingest import CANONICAL_FUNDS

_DIM = 4


class _StubModel:
    """Stand-in for SentenceTransformer that returns a fixed vector,
    so a query's "embedding" is whatever the test wants it to be."""

    def __init__(self, vector: np.ndarray):
        self._vector = vector

    def encode(self, texts, convert_to_numpy=True):
        return np.array([self._vector for _ in texts], dtype=np.float32)


def _fake_index_and_chunks():
    chunks = [
        {
            "text": "The Small Cap Fund invests predominantly in small cap "
                    "equities and carries a very high risk profile.",
            "fund_name": "Bajaj Finserv Small Cap Fund",
            "page_number": 23,
            "chunk_id": "small-cap-p23-0",
        },
        {
            "text": "The Large Cap Fund invests in blue chip large cap "
                    "companies with a moderately high risk profile.",
            "fund_name": "Bajaj Finserv Large Cap Fund",
            "page_number": 16,
            "chunk_id": "large-cap-p16-0",
        },
        {
            "text": "Expense ratio is the annual fee the fund house charges "
                    "for managing the scheme, expressed as a % of AUM.",
            "fund_name": "general",
            "page_number": 11,
            "chunk_id": "general-p11-0",
        },
    ]

    # Deterministic pseudo-embeddings: each chunk "keys" on its own basis
    # dimension, so cosine similarity between a query and a chunk is
    # predictable without needing the real embedding model.
    vectors = np.eye(len(chunks), _DIM, dtype=np.float32)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(_DIM)
    index.add(vectors)
    return index, chunks, vectors


def test_fund_name_query_applies_filter(monkeypatch):
    index, chunks, vectors = _fake_index_and_chunks()
    monkeypatch.setattr(retrieve_module, "load_index", lambda: (index, chunks))
    # Query embeds identically to the small-cap chunk's vector.
    monkeypatch.setattr(retrieve_module, "_get_model", lambda: _StubModel(vectors[0]))

    result = retrieve_module.retrieve("what is the risk of the small cap fund", top_k=3)

    assert result["found"] is True
    assert result["matched_fund_filter"] == "Bajaj Finserv Small Cap Fund"
    # Large Cap Fund chunk must be excluded by the filter even though it's
    # in the same index -- only the matched fund + general chunks qualify.
    assert all(
        chunk["fund_name"] in ("Bajaj Finserv Small Cap Fund", "general")
        for chunk in result["chunks"]
    )


def test_low_similarity_query_returns_not_found(monkeypatch):
    index, chunks, vectors = _fake_index_and_chunks()
    monkeypatch.setattr(retrieve_module, "load_index", lambda: (index, chunks))
    # A vector far from every stored chunk (no dimension overlap) -> every
    # similarity score should land below MIN_SIMILARITY_THRESHOLD.
    off_vector = np.zeros(_DIM, dtype=np.float32)
    off_vector[-1] = 1.0
    monkeypatch.setattr(retrieve_module, "_get_model", lambda: _StubModel(off_vector))

    result = retrieve_module.retrieve("does bajaj finserv offer a cryptocurrency fund", top_k=3)

    assert result == {"found": False, "chunks": [], "answer_hint": "not found in factsheet"}


@pytest.mark.parametrize("tier", list(TIER_TO_FUNDS.keys()))
def test_get_pitchable_fund_returns_valid_canonical_fund(tier):
    fund = get_pitchable_fund(tier)

    assert fund in CANONICAL_FUNDS
