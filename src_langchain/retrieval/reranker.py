"""Cross-Encoder reranker for post-RRF fine-grained relevance scoring (LangChain backend)."""

from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document

from src_langchain.config import RERANK_MODEL, RERANK_MAX_LENGTH, RERANK_CANDIDATES_K

_reranker_instance: Optional["CrossEncoderReranker"] = None


def get_reranker() -> "CrossEncoderReranker":
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance


class CrossEncoderReranker:
    """Re-score LangChain Documents with a Cross-Encoder model."""

    def __init__(
        self,
        model_name: str = RERANK_MODEL,
        max_length: int = RERANK_MAX_LENGTH,
    ):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_name, max_length=max_length, trust_remote_code=True)

    def rerank(
        self,
        query: str,
        candidates: List[Document],
        top_k: int,
    ) -> List[Document]:
        if not candidates:
            return []

        pairs = [(query, doc.page_content) for doc in candidates]

        ce_scores = self._model.predict(pairs, show_progress_bar=False)

        if not hasattr(ce_scores, "__len__"):
            ce_scores = [ce_scores]

        for doc, ce_score in zip(candidates, ce_scores):
            doc.metadata["rrf_score"] = doc.metadata.get("rrf_score", 0)
            doc.metadata["ce_score"] = float(ce_score)

        # Sort by CE score descending
        scored = list(zip(ce_scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]
