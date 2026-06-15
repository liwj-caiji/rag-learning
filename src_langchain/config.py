"""
Centralized configuration for the LangChain-based RAG system.
"""

import os

# ============================================================================
# Paths
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DISHES_DIR = os.path.join(BASE_DIR, "base", "HowToCook", "dishes")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "data", "vectorstore")

FAISS_INDEX_DIR = os.path.join(VECTORSTORE_DIR, "faiss_langchain")
CHUNKS_PATH = os.path.join(VECTORSTORE_DIR, "chunks_langchain.pkl")
BM25_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "bm25_langchain.pkl")

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

DENSE_CANDIDATES_K = 20
SPARSE_CANDIDATES_K = 20
RRF_K = 60.0

BM25_SCORE_MIN = 0.0

# ============================================================================
# Retrieval — recommendation
# ============================================================================

RECOMMEND_TOPK_DEFAULT = 5
RECOMMEND_MAX_PROBES = 3
RECOMMEND_PROBE_CANDIDATES = 15

# ============================================================================
# Retrieval — pipeline per-intent overrides
# ============================================================================

PIPELINE_HOWTO_K = 20
PIPELINE_HOWTO_DENSE_K = 50
PIPELINE_HOWTO_SPARSE_K = 50

PIPELINE_INGREDIENT_K = 20
PIPELINE_INGREDIENT_DENSE_K = 50
PIPELINE_INGREDIENT_SPARSE_K = 50

# ============================================================================
# Diversity
# ============================================================================

MMR_LAMBDA = 0.5
MMR_TOPK_DEFAULT = 5

SIM_SAME_DISH = 0.9
SIM_SAME_CATEGORY = 0.3
SIM_DIFFERENT = 0.0

# ============================================================================
# Metadata filters
# ============================================================================

CALORIE_LOW_THRESHOLD = 300
CALORIE_HIGH_THRESHOLD = 600

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
# Langfuse tracing
# ============================================================================

LANGFUSE_PUBLIC_KEY_ENV = "LANGFUSE_PUBLIC_KEY"
LANGFUSE_SECRET_KEY_ENV = "LANGFUSE_SECRET_KEY"
LANGFUSE_BASE_URL_ENV = "LANGFUSE_BASE_URL"

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

# ============================================================================
# Streaming
# ============================================================================

LLM_GEN_STREAM = True
LLM_GEN_STREAM_TIMEOUT = 30.0

# ============================================================================
# Cross-Encoder Reranker
# ============================================================================

RERANK_ENABLED = True
RERANK_MODEL = "BAAI/bge-reranker-v2-minicpm"
RERANK_CANDIDATES_K = 30    # candidates fed into Cross-Encoder from RRF top-N
RERANK_MAX_LENGTH = 512      # max token length for CrossEncoder input
