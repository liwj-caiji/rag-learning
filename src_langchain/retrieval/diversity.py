"""Diversity reranking for multi-dish recommendation."""

from typing import Dict, List

import numpy as np
from langchain_core.documents import Document

from ..config import MMR_TOPK_DEFAULT, MMR_LAMBDA, SIM_SAME_DISH, SIM_SAME_CATEGORY, SIM_DIFFERENT


def diversify_by_category(
    documents: List[Document],
    k: int = MMR_TOPK_DEFAULT,
) -> List[Document]:
    """Round-robin selection across categories."""
    if not documents:
        return []

    groups: Dict[str, List[Document]] = {}
    for doc in documents:
        cat = doc.metadata.get("category", "unknown")
        groups.setdefault(cat, []).append(doc)

    result = []
    iterators = {cat: iter(g) for cat, g in groups.items()}
    cats = list(groups.keys())

    while len(result) < k and any(
        iterators[cat] is not None for cat in cats
    ):
        for cat in cats:
            it = iterators[cat]
            if it is None:
                continue
            try:
                result.append(next(it))
                if len(result) >= k:
                    break
            except StopIteration:
                iterators[cat] = None

    return result[:k]


def mmr_rerank_documents(
    documents: List[Document],
    lambda_: float = MMR_LAMBDA,
    k: int = MMR_TOPK_DEFAULT,
) -> List[Document]:
    """MMR reranking using metadata-based similarity."""
    if not documents or k <= 0:
        return []

    n = len(documents)
    scores = np.array([
        doc.metadata.get("rrf_score", doc.metadata.get("bm25_score", 0.0))
        for doc in documents
    ], dtype=np.float32)

    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    selected: List[int] = []
    remaining = list(range(n))

    for _ in range(min(k, n)):
        mmr_scores = []
        for i in remaining:
            rel = scores[i] if lambda_ < 1.0 else 1.0
            if selected:
                sim_to_selected = max(
                    _doc_similarity(documents[i], documents[j])
                    for j in selected
                )
                div = 1.0 - sim_to_selected
            else:
                div = 1.0
            mmr_scores.append(lambda_ * rel + (1 - lambda_) * div)

        best_idx = remaining[np.argmax(mmr_scores)]
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [documents[i] for i in selected]


def _doc_similarity(a: Document, b: Document) -> float:
    ma = a.metadata
    mb = b.metadata
    if ma.get("dish_name") and ma["dish_name"] == mb.get("dish_name"):
        return SIM_SAME_DISH
    if ma.get("category") and ma["category"] == mb.get("category"):
        return SIM_SAME_CATEGORY
    return SIM_DIFFERENT
