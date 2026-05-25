"""Build FAISS / BM25 indices from recipe markdown files."""

import os
import pickle
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    VECTORSTORE_DIR, FAISS_INDEX_PATH, CHUNKS_PATH, BM25_INDEX_PATH,
    EMBED_MODEL,
)
from .config import chinese_tokenize
from .splitter import RecipeSplitter, collect_all_recipes


def build_index() -> Tuple[int, int]:
    """Build FAISS index from all recipes and persist to disk.

    Returns:
        (num_documents, num_chunks)
    """
    import faiss

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    print("Collecting recipes...")
    recipe_paths = collect_all_recipes()
    print(f"  Found {len(recipe_paths)} recipe files")

    print("Splitting recipes...")
    all_chunks: List[Dict] = []
    for i, path in enumerate(recipe_paths):
        splitter = RecipeSplitter(path)
        chunks = splitter.split()
        all_chunks.extend(chunks)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(recipe_paths)} files, "
                  f"{len(all_chunks)} chunks so far")

    print(f"  Total chunks: {len(all_chunks)}")

    texts = [c["text"] for c in all_chunks]
    print(f"Encoding {len(texts)} chunks with {EMBED_MODEL}...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)

    # L2-normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Persist
    print(f"Saving FAISS index to {FAISS_INDEX_PATH}")
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Build and persist BM25 index
    print(f"Building BM25 index...")
    from rank_bm25 import BM25Okapi

    tokenized = [chinese_tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized)
    bm25_payload = {
        "bm25": bm25,
        "tokenized_corpus": tokenized,
        "texts": texts,
    }
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_payload, f)
    print(f"  BM25 index saved to {BM25_INDEX_PATH}")

    # Save chunks metadata alongside the index
    # Convert to a pickle-friendly format (strip model internals)
    serializable = []
    for c in all_chunks:
        serializable.append({
            "text": c["text"],
            "level": c["level"],
            "metadata": dict(c["metadata"]),
        })
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(serializable, f)

    # Count unique dishes
    dish_names = set()
    for c in all_chunks:
        dn = c["metadata"].get("dish_name")
        if dn:
            dish_names.add(dn)

    print(f"\nDone! Summary:")
    print(f"  Recipes (dishes): {len(dish_names)}")
    print(f"  Total chunks:     {len(all_chunks)}")
    print(f"  Embedding dim:    {dim}")
    print(f"  FAISS index:      {FAISS_INDEX_PATH}")
    print(f"  BM25 index:       {BM25_INDEX_PATH}")
    print(f"  Chunks path:      {CHUNKS_PATH}")

    return len(dish_names), len(all_chunks)


def load_index() -> Tuple[object, List[Dict]]:
    """Load FAISS index and chunks from disk.

    Returns:
        (faiss_index, chunks)
    """
    import faiss

    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"Index not found at {FAISS_INDEX_PATH}. "
            f"Run `python -m src.preprocess.indexer` first."
        )

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def search(query: str, k: int = 5) -> List[Dict]:
    """Quick demo: embed a query and retrieve top-k chunks."""
    index, chunks = load_index()

    model = SentenceTransformer(EMBED_MODEL)
    q_emb = model.encode([query]).astype(np.float32)

    import faiss
    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "score": float(score),
            "chunk": chunks[idx],
        })
    return results


if __name__ == "__main__":
    build_index()
