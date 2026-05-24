import os
import pickle
from typing import List, Dict, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from ..preprocess.config import VECTORSTORE_DIR, EMBED_MODEL, chinese_tokenize

# Paths to pre-built index files
FAISS_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(VECTORSTORE_DIR, "chunks.pkl")
BM25_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "bm25_index.pkl")

_RETRIEVAL_CACHE = {}


def _load_chunks() -> List[Dict]:
    if "chunks" not in _RETRIEVAL_CACHE:
        with open(CHUNKS_PATH, "rb") as f:
            _RETRIEVAL_CACHE["chunks"] = pickle.load(f)
    return _RETRIEVAL_CACHE["chunks"]


def _load_faiss():
    import faiss

    if "faiss" not in _RETRIEVAL_CACHE:
        _RETRIEVAL_CACHE["faiss"] = faiss.read_index(FAISS_INDEX_PATH)
    return _RETRIEVAL_CACHE["faiss"]


def _load_bm25():
    if "bm25_payload" not in _RETRIEVAL_CACHE:
        with open(BM25_INDEX_PATH, "rb") as f:
            _RETRIEVAL_CACHE["bm25_payload"] = pickle.load(f)
    return _RETRIEVAL_CACHE["bm25_payload"]


def _get_model():
    if "model" not in _RETRIEVAL_CACHE:
        _RETRIEVAL_CACHE["model"] = SentenceTransformer(EMBED_MODEL)
    return _RETRIEVAL_CACHE["model"]


def dense_search(query: str, k: int = 50) -> List[Dict]:
    """Dense retrieval via FAISS cosine similarity.

    Returns list of {score, chunk} sorted descending by score.
    """
    import faiss

    index = _load_faiss()
    chunks = _load_chunks()
    model = _get_model()

    q_emb = model.encode([query]).astype(np.float32)
    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({"score": float(score), "chunk": chunks[idx]})
    return results


def sparse_search(query: str, k: int = 50) -> List[Dict]:
    """Sparse retrieval via BM25 keyword matching.

    Returns list of {score, chunk} sorted descending by score.
    """
    payload = _load_bm25()
    bm25 = payload["bm25"]
    chunks = _load_chunks()

    tokenized_query = chinese_tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        results.append({"score": float(scores[idx]), "chunk": chunks[idx]})
    return results


def hybrid_search(
    query: str,
    k: int = 5,
    dense_k: int = 50,
    sparse_k: int = 50,
    rrf_k: float = 60.0,
) -> List[Dict]:
    """Hybrid search combining dense (FAISS) and sparse (BM25) with RRF fusion.

    Parameters:
        query:    User query text.
        k:        Number of final results to return.
        dense_k:  Candidates per channel, default 50.
        sparse_k: Candidates per channel, default 50.
        rrf_k:    RRF constant, default 60.0 (classic value).

    Returns:
        List of {score (RRF), dense_score, sparse_score, chunk} sorted by RRF score.
    """
    # Run both channels
    dense_results = dense_search(query, k=dense_k)
    sparse_results = sparse_search(query, k=sparse_k)

    # Build rank lookup: chunk_index -> rank (1-based)
    dense_ranks: Dict[int, int] = {
        id(r["chunk"]): rank + 1
        for rank, r in enumerate(dense_results)
    }
    sparse_ranks: Dict[int, int] = {
        id(r["chunk"]): rank + 1
        for rank, r in enumerate(sparse_results)
    }

    # Collect all unique chunk IDs
    all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

    # Compute RRF score for each chunk
    fused = []
    for cid in all_ids:
        rrf_score = 0.0
        if cid in dense_ranks:
            rrf_score += 1.0 / (rrf_k + dense_ranks[cid])
        if cid in sparse_ranks:
            rrf_score += 1.0 / (rrf_k + sparse_ranks[cid])

        # Retrieve the chunk object
        chunk = None
        for r in dense_results:
            if id(r["chunk"]) == cid:
                chunk = r["chunk"]
                break
        if chunk is None:
            for r in sparse_results:
                if id(r["chunk"]) == cid:
                    chunk = r["chunk"]
                    break

        fused.append({
            "score": rrf_score,
            "dense_rank": dense_ranks.get(cid),
            "sparse_rank": sparse_ranks.get(cid),
            "chunk": chunk,
        })

    # Sort by RRF score descending
    fused.sort(key=lambda x: x["score"], reverse=True)

    return fused[:k]
