"""
Centralized configuration for the RAG system.

All tunable constants live here. Import directly into any module:

    from src.config import HYBRID_TOPK_DEFAULT, LLM_GEN_MODEL, ...
"""

import os

# ============================================================================
# Paths
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DISHES_DIR = os.path.join(BASE_DIR, "base", "HowToCook", "dishes")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "data", "vectorstore")

FAISS_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(VECTORSTORE_DIR, "chunks.pkl")
BM25_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "bm25_index.pkl")

LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "rag.log")

# ============================================================================
# Embedding
# ============================================================================

EMBED_MODEL = "shibing624/text2vec-base-chinese"
EMBED_DIM = 768

# ============================================================================
# Chunking
# ============================================================================

SKIP_DIRS = {"template"}
REMOVE_FOOTER = True
SPLIT_H3 = True

# ============================================================================
# Retrieval — hybrid search
# ============================================================================

HYBRID_TOPK_MIN = 1
HYBRID_TOPK_MAX = 20
HYBRID_TOPK_DEFAULT = 5

DENSE_CANDIDATES_K = 20     # candidates from FAISS channel
SPARSE_CANDIDATES_K = 20    # candidates from BM25 channel
RRF_K = 60.0                # RRF fusion constant

BM25_SCORE_MIN = 0.0        # drop BM25 results with score <= this

# ============================================================================
# Retrieval — recommendation
# ============================================================================

RECOMMEND_TOPK_DEFAULT = 5
RECOMMEND_MAX_PROBES = 3          # max search probes to iterate
RECOMMEND_PROBE_CANDIDATES = 15   # candidates collected per probe

# ============================================================================
# Retrieval — pipeline per-intent overrides (howto / ingredient)
# ============================================================================

PIPELINE_HOWTO_K = 20
PIPELINE_HOWTO_DENSE_K = 50
PIPELINE_HOWTO_SPARSE_K = 50

PIPELINE_INGREDIENT_K = 20
PIPELINE_INGREDIENT_DENSE_K = 50
PIPELINE_INGREDIENT_SPARSE_K = 50

# ============================================================================
# Diversity (MMR & category round-robin)
# ============================================================================

MMR_LAMBDA = 0.5              # 1.0 = pure relevance, 0.0 = pure diversity
MMR_TOPK_DEFAULT = 5

# Similarity heuristics (embeddings unavailable path)
SIM_SAME_DISH = 0.9
SIM_SAME_CATEGORY = 0.3
SIM_DIFFERENT = 0.0

# ============================================================================
# Metadata filters
# ============================================================================

CALORIE_LOW_THRESHOLD = 300    # kcal <= this → "low"
CALORIE_HIGH_THRESHOLD = 600   # kcal >= this → "high"

# ============================================================================
# LLM — Intent classification
# ============================================================================

LLM_INTENT_MODEL = "deepseek-v4-flash"
LLM_INTENT_API_BASE = "https://api.deepseek.com"
LLM_INTENT_TIMEOUT = 8.0
LLM_INTENT_TEMPERATURE = 0.1
LLM_INTENT_MAX_TOKENS = 512

# ============================================================================
# LLM — Response generation
# ============================================================================

LLM_GEN_MODEL = "deepseek-v4-flash"
LLM_GEN_API_BASE = "https://api.deepseek.com"
LLM_GEN_TIMEOUT = 15.0
LLM_GEN_TEMPERATURE = 0.3
LLM_GEN_MAX_TOKENS = 2048

# ============================================================================
# LLM — Shared
# ============================================================================

LLM_API_KEY_ENV = "DEEPSEEK_API_KEY"

# ============================================================================
# App (Gradio server)
# ============================================================================

APP_HOST = "127.0.0.1"
APP_PORT = 7860
APP_QUEUE_DEFAULT_CONCURRENCY = 5
APP_QUEUE_MAX_SIZE = 20
APP_EVENT_CONCURRENCY = 3

# ============================================================================
# Logging
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"
LOG_SUPPRESS = ("httpx", "faiss", "sentence_transformers", "huggingface_hub", "transformers")
