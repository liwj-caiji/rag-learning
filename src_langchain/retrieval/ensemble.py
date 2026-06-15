"""Hybrid retrieval with Reciprocal Rank Fusion (RRF).

Custom BaseRetriever subclass replacing LangChain's EnsembleRetriever
(which only supports weighted sum, not RRF).
"""

import os
import pickle
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from ..config import (
    FAISS_INDEX_DIR, EMBED_MODEL, CHUNKS_PATH, BM25_INDEX_PATH,
    HYBRID_TOPK_DEFAULT, DENSE_CANDIDATES_K, SPARSE_CANDIDATES_K, RRF_K,
    RECOMMEND_TOPK_DEFAULT, RECOMMEND_MAX_PROBES, RECOMMEND_PROBE_CANDIDATES,
    MMR_LAMBDA, MMR_TOPK_DEFAULT,
)
from .sparse_retriever import SparseRetriever
from .diversity import diversify_by_category, mmr_rerank_documents
from .filters import apply_filters

_RETRIEVAL_CACHE = {}


def _load_chunks() -> List[Document]:
    if "chunks" not in _RETRIEVAL_CACHE:
        with open(CHUNKS_PATH, "rb") as f:
            _RETRIEVAL_CACHE["chunks"] = pickle.load(f)
    return _RETRIEVAL_CACHE["chunks"]


def _load_faiss() -> FAISS:
    if "faiss" not in _RETRIEVAL_CACHE:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        _RETRIEVAL_CACHE["faiss"] = FAISS.load_local(
            FAISS_INDEX_DIR, embeddings,
            allow_dangerous_deserialization=True,
        )
    return _RETRIEVAL_CACHE["faiss"]


class RRFFusionRetriever(BaseRetriever):
    """Hybrid retriever: dense (FAISS) + sparse (BM25) with RRF fusion."""

    k: int = HYBRID_TOPK_DEFAULT
    dense_k: int = DENSE_CANDIDATES_K
    sparse_k: int = SPARSE_CANDIDATES_K
    rrf_k: float = RRF_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        # Dense channel
        faiss_store = _load_faiss()
        faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": self.dense_k})
        dense_docs = faiss_retriever.invoke(query)
        for rank, doc in enumerate(dense_docs):
            doc.metadata["dense_rank"] = rank + 1

        # Sparse channel
        sparse_retriever = SparseRetriever(k=self.sparse_k)
        sparse_docs = sparse_retriever.invoke(query)
        for rank, doc in enumerate(sparse_docs):
            doc.metadata["sparse_rank"] = rank + 1

        # Build rank lookup
        dense_ranks: Dict[int, int] = {
            id(doc): rank + 1 for rank, doc in enumerate(dense_docs)
        }
        sparse_ranks: Dict[int, int] = {
            id(doc): rank + 1 for rank, doc in enumerate(sparse_docs)
        }

        # Collect unique documents
        seen = {}
        for doc in dense_docs:
            seen[id(doc)] = doc
        for doc in sparse_docs:
            if id(doc) not in seen:
                seen[id(doc)] = doc

        # RRF scoring
        fused: List[tuple] = []
        for cid, doc in seen.items():
            rrf_score = 0.0
            if cid in dense_ranks:
                rrf_score += 1.0 / (self.rrf_k + dense_ranks[cid])
            if cid in sparse_ranks:
                rrf_score += 1.0 / (self.rrf_k + sparse_ranks[cid])
            doc.metadata["rrf_score"] = rrf_score
            doc.metadata["dense_rank"] = dense_ranks.get(cid)
            doc.metadata["sparse_rank"] = sparse_ranks.get(cid)
            fused.append((rrf_score, doc))

        fused.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in fused[:self.k]]


def hybrid_search(
    query: str,
    k: int = HYBRID_TOPK_DEFAULT,
    dense_k: int = DENSE_CANDIDATES_K,
    sparse_k: int = SPARSE_CANDIDATES_K,
    rrf_k: float = RRF_K,
    rerank: bool = False,
    rerank_top_n: int = 30,
) -> List[Document]:
    retriever = RRFFusionRetriever(
        k=(rerank_top_n if rerank else k),
        dense_k=dense_k, sparse_k=sparse_k, rrf_k=rrf_k,
    )
    results = retriever.invoke(query)

    if rerank:
        from .reranker import get_reranker
        results = get_reranker().rerank(query, results, top_k=k)

    return results


def recommend_dishes(
    query: str,
    k: int = RECOMMEND_TOPK_DEFAULT,
    filters: Optional[Dict] = None,
    diversify: bool = True,
    probes: Optional[List[str]] = None,
) -> List[Document]:
    """Multi-dish recommendation using multi-probe hybrid search."""
    candidates: List[Document] = []
    seen_dishes: set = set()

    search_queries = (probes or []) + [query]
    for sq in search_queries[:RECOMMEND_MAX_PROBES]:
        results = hybrid_search(sq, k=RECOMMEND_PROBE_CANDIDATES)
        for doc in results:
            if doc.metadata.get("level") != "dish":
                continue
            dish_name = doc.metadata.get("dish_name", "")
            if dish_name in seen_dishes:
                continue
            seen_dishes.add(dish_name)
            candidates.append(doc)

    if not candidates:
        return []

    if filters:
        candidates = apply_filters(candidates, filters)

    if not candidates:
        return []

    if diversify and len(candidates) > 1:
        try:
            ranked = diversify_by_category(candidates, k=k)
            return ranked[:k]
        except Exception:
            pass

    candidates.sort(
        key=lambda x: x.metadata.get("rrf_score", 0), reverse=True
    )
    return candidates[:k]
