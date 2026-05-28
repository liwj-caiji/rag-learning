"""Sparse retrieval via BM25 with Chinese tokenization."""

import pickle
from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from ..config import BM25_INDEX_PATH, SPARSE_CANDIDATES_K
from shared.tokenizer import chinese_tokenize

_BM25_CACHE = {}


def _load_bm25_payload():
    if "payload" not in _BM25_CACHE:
        with open(BM25_INDEX_PATH, "rb") as f:
            _BM25_CACHE["payload"] = pickle.load(f)
    return _BM25_CACHE["payload"]


class SparseRetriever(BaseRetriever):
    """BM25 sparse retriever for Chinese text."""

    k: int = SPARSE_CANDIDATES_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        payload = _load_bm25_payload()
        bm25 = payload["retriever"]
        bm25.k = self.k
        results = bm25.invoke(query)
        for doc in results:
            if "bm25_score" not in doc.metadata:
                doc.metadata["bm25_score"] = doc.metadata.get("score", 0.0)
        return results


def sparse_search(query: str, k: int = SPARSE_CANDIDATES_K) -> List[Document]:
    return SparseRetriever(k=k).invoke(query)
