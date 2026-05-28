"""Backward-compat re-exports; prefer ``from src.config import ...`` in new code."""

from src.config import (  # noqa: F401
    DISHES_DIR,
    VECTORSTORE_DIR,
    SKIP_DIRS,
    EMBED_MODEL,
    REMOVE_FOOTER,
    SPLIT_H3,
)

from shared.tokenizer import chinese_tokenize  # noqa: F401
