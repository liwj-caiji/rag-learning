"""Cross-Encoder reranker for post-RRF fine-grained relevance scoring."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.config import RERANK_MODEL, RERANK_MAX_LENGTH, RERANK_CANDIDATES_K

_reranker_instance: Optional["CrossEncoderReranker"] = None


def get_reranker() -> "CrossEncoderReranker":
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance


class CrossEncoderReranker:
    """Re-score candidates with a Cross-Encoder model.

    Uses BAAI/bge-reranker-v2-m3, a BERT-based multilingual reranker
    that captures fine-grained semantic interaction via full
    self-attention across (query, chunk) pairs.
    """

    def __init__(
        self,
        model_name: str = RERANK_MODEL,
        max_length: int = RERANK_MAX_LENGTH,
    ):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_name, max_length=max_length)

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        if not candidates:
            return []

        pairs = [
            (query, c["chunk"].get("text", ""))
            for c in candidates
        ]

        ce_scores = self._model.predict(pairs, show_progress_bar=False)

        if not hasattr(ce_scores, "__len__"):
            ce_scores = [ce_scores]

        for c, ce_score in zip(candidates, ce_scores):
            c["rrf_score"] = c["score"]
            c["ce_score"] = float(ce_score)
            c["score"] = float(ce_score)

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]
