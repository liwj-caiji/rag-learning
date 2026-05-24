import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DISHES_DIR = os.path.join(BASE_DIR, "base", "HowToCook", "dishes")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "data", "vectorstore")

# Directories to skip
SKIP_DIRS = {"template"}

# Embedding model
EMBED_MODEL = "shibing624/text2vec-base-chinese"

# Chunking settings
REMOVE_FOOTER = True          # Remove the standard PR footer
SPLIT_H3 = True               # Split ## 操作 into ### sub-sections

# Shared Chinese tokenizer for BM25
import jieba
import re

# Use precise mode by default
jieba.setLogLevel(20)  # Suppress INFO logs


def chinese_tokenize(text: str) -> list[str]:
    """Tokenize text for BM25: jieba for Chinese, preserve English/numbers."""
    tokens = jieba.lcut(text)
    # Remove whitespace-only tokens
    return [t.strip() for t in tokens if t.strip()]
