"""Build and persist FAISS / BM25 indices using LangChain components."""

import os
import pickle
from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from ..config import (
    VECTORSTORE_DIR, FAISS_INDEX_DIR, CHUNKS_PATH, BM25_INDEX_PATH,
    EMBED_MODEL,
)
from .loader import load_recipe_documents
from .splitter import RecipeDocumentTransformer
from shared.tokenizer import chinese_tokenize


def build_index() -> tuple:
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    print("Loading recipe documents...")
    raw_docs = load_recipe_documents()
    print(f"  Found {len(raw_docs)} recipe files")

    print("Splitting into chunks...")
    transformer = RecipeDocumentTransformer()
    chunks = transformer.transform_documents(raw_docs)
    print(f"  Total chunks: {len(chunks)}")

    print(f"Building FAISS index with {EMBED_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings, distance_strategy="COSINE")
    print(f"Saving FAISS index to {FAISS_INDEX_DIR}")
    vectorstore.save_local(FAISS_INDEX_DIR)

    print("Building BM25 index...")
    bm25_retriever = BM25Retriever.from_documents(
        chunks,
        preprocess_func=chinese_tokenize,
    )
    bm25_payload = {"retriever": bm25_retriever, "documents": chunks}
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_payload, f)
    print(f"  BM25 index saved to {BM25_INDEX_PATH}")

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    dish_names = set()
    for c in chunks:
        dn = c.metadata.get("dish_name")
        if dn:
            dish_names.add(dn)

    print(f"\nDone! Summary:")
    print(f"  Recipes (dishes): {len(dish_names)}")
    print(f"  Total chunks:     {len(chunks)}")
    print(f"  FAISS index:      {FAISS_INDEX_DIR}")
    print(f"  BM25 index:       {BM25_INDEX_PATH}")
    print(f"  Chunks path:      {CHUNKS_PATH}")

    return len(dish_names), len(chunks)


def load_faiss_index() -> FAISS:
    if not os.path.exists(FAISS_INDEX_DIR):
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_DIR}. "
            f"Run `python -m src_langchain.preprocess.indexer` first."
        )
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(
        FAISS_INDEX_DIR, embeddings,
        allow_dangerous_deserialization=True,
    )


def load_chunks() -> List[Document]:
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"Chunks file not found at {CHUNKS_PATH}. "
            f"Run `python -m src_langchain.preprocess.indexer` first."
        )
    with open(CHUNKS_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    build_index()
