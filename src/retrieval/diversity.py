"""Diversity reranking for multi-dish recommendation."""

from typing import Dict, List, Optional

import numpy as np


def diversify_by_category(
    chunks: List[Dict],
    k: int = 5,
) -> List[Dict]:
    """Round-robin selection across categories.

    Groups chunks by category, then picks one from each category
    in round-robin until k is reached.
    """
    if not chunks:
        return []

    # Group by category
    groups: Dict[str, List[Dict]] = {}
    for c in chunks:
        cat = c.get("metadata", {}).get("category", "unknown")
        groups.setdefault(cat, []).append(c)

    # Round-robin selection
    result = []
    iters = {cat: iter(g) for cat, g in groups.items()}
    cats = list(groups.keys())

    while len(result) < k and any(
        iters[cat] is not None for cat in cats
    ):
        for cat in cats:
            it = iters[cat]
            if it is None:
                continue
            try:
                result.append(next(it))
                if len(result) >= k:
                    break
            except StopIteration:
                iters[cat] = None

    return result[:k]


def mmr_rerank(
    chunks: List[Dict],
    query_emb: np.ndarray,
    lambda_: float = 0.5,
    k: int = 5,
) -> List[Dict]:
    """Maximum Marginal Relevance reranking.

    Balances query relevance (score) with diversity (dissimilarity to selected).

    Args:
        chunks: List of chunks, each must have a 'score' field.
        query_emb: Query embedding vector (1D array).
        lambda_: Trade-off parameter. 1.0 = pure relevance, 0.0 = pure diversity.
        k: Number of results to return.

    Returns:
        Reranked list of chunks.
    """
    if not chunks or k <= 0:
        return []

    n = len(chunks)
    # Build similarity matrix from chunk scores (for relevance) and
    # use cosine distance for diversity
    scores = np.array([c.get("score", 0.0) for c in chunks], dtype=np.float32)
    # Normalize scores to [0, 1]
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    selected = []
    remaining = list(range(n))

    for _ in range(min(k, n)):
        mmr_scores = []
        for i in remaining:
            # Relevance term
            rel = scores[i] if lambda_ < 1.0 else 1.0

            # Diversity term: max similarity to already selected
            if selected:
                sim_to_selected = max(
                    _chunk_sim(chunks[i], chunks[j])
                    for j in selected
                )
                div = 1.0 - sim_to_selected
            else:
                div = 1.0

            mmr = lambda_ * rel + (1 - lambda_) * div
            mmr_scores.append(mmr)

        # Pick best
        best_idx = remaining[np.argmax(mmr_scores)]
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [chunks[i] for i in selected]


def _chunk_sim(a: Dict, b: Dict) -> float:
    """Estimate similarity between two chunks.

    Uses metadata overlap (dish_name, category, section_type) as a proxy
    when embeddings are not available.
    """
    ma = a.get("metadata", {})
    mb = b.get("metadata", {})

    # Same dish → very similar
    if ma.get("dish_name") and ma["dish_name"] == mb.get("dish_name"):
        return 0.9
    # Same category → somewhat similar
    if ma.get("category") and ma["category"] == mb.get("category"):
        return 0.3
    return 0.0
