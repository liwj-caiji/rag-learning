"""Backward-compat re-exports; prefer ``from src.config import ...`` in new code."""

from src.config import (  # noqa: F401
    DISHES_DIR,
    VECTORSTORE_DIR,
    SKIP_DIRS,
    EMBED_MODEL,
    REMOVE_FOOTER,
    SPLIT_H3,
)

import jieba
import re

jieba.setLogLevel(20)


def chinese_tokenize(text: str) -> list[str]:
    """Tokenize text for BM25: jieba for Chinese, preserve English/numbers."""
    tokens = jieba.lcut(text)
    return [t.strip() for t in tokens if t.strip()]
