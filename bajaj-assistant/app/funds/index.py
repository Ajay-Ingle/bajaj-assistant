"""Builds/loads the FAISS index.

Run as a script (`python -m app.funds.index`) to (re)build the index from
the factsheet PDF end to end — this file doubles as the build step.
"""

import faiss
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import FACTSHEET_PDF_PATH, FUND_CHUNKS_PATH, FUND_FAISS_INDEX_PATH
from app.funds.ingest import build_chunks

# Small, fast, runs locally -- no API key needed.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def build_index(chunks: list[dict]) -> None:
    """Embed chunk texts, build a cosine-similarity FAISS index, save both
    the index and the chunk metadata (everything except the raw vectors)
    to data/processed/."""
    if not chunks:
        raise ValueError("build_index() called with no chunks — nothing to index.")

    model = _get_model()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Normalize so inner product == cosine similarity, same pattern as
    # standard sentence-embedding RAG setups.
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    FUND_FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FUND_FAISS_INDEX_PATH))
    joblib.dump(chunks, FUND_CHUNKS_PATH)

    print(f"built FAISS index with {index.ntotal} vectors (dim={dim})")
    print(f"saved index to {FUND_FAISS_INDEX_PATH}")
    print(f"saved chunk metadata to {FUND_CHUNKS_PATH}")


def load_index():
    """Load the FAISS index + chunk metadata built by build_index().

    Raises a clear error (not a stack trace) if the build step hasn't been
    run yet.
    """
    if not FUND_FAISS_INDEX_PATH.exists() or not FUND_CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Fund index not found at {FUND_FAISS_INDEX_PATH} and/or "
            f"{FUND_CHUNKS_PATH}. Run `python -m app.funds.index` to build "
            f"it first."
        )
    index = faiss.read_index(str(FUND_FAISS_INDEX_PATH))
    chunks = joblib.load(FUND_CHUNKS_PATH)
    return index, chunks


if __name__ == "__main__":
    built_chunks = build_chunks(str(FACTSHEET_PDF_PATH))
    build_index(built_chunks)
