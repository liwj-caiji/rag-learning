"""Load recipe markdown files using LangChain document loaders."""

import os
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader

from ..config import DISHES_DIR, SKIP_DIRS


def _is_recipe_file(path: str) -> bool:
    if not path.endswith(".md"):
        return False
    rel = os.path.relpath(path, DISHES_DIR)
    parts = rel.replace("\\", "/").split("/")
    return not any(p in SKIP_DIRS for p in parts)


def load_recipe_documents(dishes_dir: str = DISHES_DIR) -> List[Document]:
    loader = DirectoryLoader(
        dishes_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        recursive=True,
        show_progress=True,
    )

    docs = loader.load()

    filtered: List[Document] = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        if not _is_recipe_file(source):
            continue
        rel = os.path.relpath(source, dishes_dir)
        parts = rel.replace("\\", "/").split("/")
        doc.metadata["category"] = parts[0] if len(parts) > 1 else "unknown"
        filtered.append(doc)

    return filtered
