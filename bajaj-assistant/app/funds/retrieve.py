"""Retrieval function used at query time.

Retrieves grounded chunks from the factsheet only -- does NOT call Claude
to generate an answer. That wiring happens in Stage 4; this module's job
ends at returning the relevant chunks (or a clear "not found" signal).
"""

import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.funds.index import EMBEDDING_MODEL_NAME, load_index
from app.funds.ingest import CANONICAL_FUNDS, GENERAL_LABEL

# Empirically tunable -- start conservative, adjust after manual testing
# (see scripts/test_rag.py) against real queries. Below this, a match is
# treated as "not actually grounded in the factsheet" and refused rather
# than passed through as a weak/likely-wrong answer.
MIN_SIMILARITY_THRESHOLD = 0.35

# LEXICAL GROUNDING GUARD -- discovered necessary by testing the spec's own
# refusal-path example ("does bajaj finserv offer a cryptocurrency fund"):
# the factsheet's Performance-section footnotes densely repeat "Mr. X also
# manages Bajaj Finserv Y Fund" across many pages. That boilerplate shares
# so much surface vocabulary ("Bajaj Finserv", "Fund") with almost any
# query about a Bajaj Finserv fund that its cosine similarity lands at
# ~0.65-0.70 -- above MIN_SIMILARITY_THRESHOLD -- even for a fund category
# ("cryptocurrency") that never appears anywhere in the document. Cosine
# similarity alone can't be trusted to catch this; we additionally require
# at least one substantive (non-stopword) query term to literally appear
# in the top-scoring chunk's text before accepting a match as grounded.
_STOPWORDS = {
    "the", "a", "an", "of", "in", "is", "are", "what", "does", "do", "for",
    "and", "to", "on", "bajaj", "finserv", "fund", "funds", "offer",
    "offers", "provide", "provides", "have", "has", "with", "about", "any",
}


def _has_lexical_support(query: str, text: str) -> bool:
    """True if at least one substantive query term appears in `text`.

    A cheap grounding check layered on top of embedding similarity -- see
    the LEXICAL GROUNDING GUARD note above for why this is needed. If the
    query has no substantive terms at all (e.g. it's only stopwords), we
    trust the embedding score alone rather than block everything.
    """
    query_terms = [
        term for term in re.findall(r"[a-z0-9]+", query.lower())
        if len(term) > 3 and term not in _STOPWORDS
    ]
    if not query_terms:
        return True
    text_lower = text.lower()
    return any(term in text_lower for term in query_terms)


_NOT_FOUND_RESULT = {"found": False, "chunks": [], "answer_hint": "not found in factsheet"}

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _match_fund_filter(query: str) -> str | None:
    """Case-insensitive substring match of a fund name in the query.

    Lightweight hybrid-retrieval step: exact identifier matching before
    vector search -- same principle as BM25-for-exact-IDs, simplified to
    substring matching since fund names are the only "identifiers" here.
    Matches against the full canonical name and a couple of shortened
    forms (dropping the "Bajaj Finserv" prefix and "Fund"/"ETF" suffix) so
    "small cap fund" and even bare "small cap" both match "Bajaj Finserv
    Small Cap Fund". Picks the longest (most specific) match if more than
    one fund's short form appears in the query.
    """
    q = query.lower()
    best_match = None
    best_len = 0

    for fund_name in CANONICAL_FUNDS:
        lower_name = fund_name.lower()
        short = lower_name.replace("bajaj finserv", "").strip()
        bare = re.sub(r"\s+(fund|etf)(\s*-\s*growth)?$", "", short).strip()

        for candidate in (lower_name, short, bare):
            if candidate and candidate in q and len(candidate) > best_len:
                best_match = fund_name
                best_len = len(candidate)

    return best_match


def retrieve(query: str, top_k: int = 4) -> dict:
    """Retrieve the top_k most relevant factsheet chunks for `query`.

    If the query names a specific fund, search is restricted to that
    fund's chunks plus "general" chunks (a question might mix a specific
    fund with a general term). Otherwise searches the full index. Returns
    found=False if the best match is below MIN_SIMILARITY_THRESHOLD,
    instead of passing weak matches through.
    """
    index, chunks = load_index()
    if index.ntotal == 0 or not chunks:
        return dict(_NOT_FOUND_RESULT)

    matched_fund = _match_fund_filter(query)

    model = _get_model()
    query_vec = np.asarray(model.encode([query], convert_to_numpy=True), dtype=np.float32)
    faiss.normalize_L2(query_vec)

    if matched_fund:
        candidate_indices = [
            i for i, chunk in enumerate(chunks)
            if chunk["fund_name"] == matched_fund or chunk["fund_name"] == GENERAL_LABEL
        ]
        if not candidate_indices:
            return dict(_NOT_FOUND_RESULT)

        # Restrict the search space to just the candidate chunks, so the
        # fund filter is actually enforced rather than a re-ranking hint.
        sub_vectors = np.vstack(
            [index.reconstruct(i) for i in candidate_indices]
        ).astype(np.float32)
        sub_index = faiss.IndexFlatIP(sub_vectors.shape[1])
        sub_index.add(sub_vectors)

        k = min(top_k, len(candidate_indices))
        scores, sub_hits = sub_index.search(query_vec, k)
        hit_indices = [candidate_indices[h] for h in sub_hits[0]]
        hit_scores = scores[0]
    else:
        k = min(top_k, index.ntotal)
        scores, hits = index.search(query_vec, k)
        hit_indices = hits[0]
        hit_scores = scores[0]

    if len(hit_indices) == 0 or hit_scores[0] < MIN_SIMILARITY_THRESHOLD:
        return dict(_NOT_FOUND_RESULT)

    top_chunk_text = chunks[hit_indices[0]]["text"]
    if not _has_lexical_support(query, top_chunk_text):
        return dict(_NOT_FOUND_RESULT)

    result_chunks = []
    for idx, score in zip(hit_indices, hit_scores):
        chunk = dict(chunks[idx])
        chunk["similarity_score"] = float(score)
        result_chunks.append(chunk)

    return {
        "found": True,
        "chunks": result_chunks,
        "matched_fund_filter": matched_fund,
    }
