"""Hierarchical markdown chunking with LangChain's MarkdownHeaderTextSplitter."""

import os
import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from ..config import DISHES_DIR, REMOVE_FOOTER, SPLIT_H3

FOOTER_RE = re.compile(
    r"如果您遵循本指南的制作流程而发现有问题或可以改进的流程，请提出 Issue 或 Pull request 。"
)

HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
]


class RecipeDocumentTransformer:
    """Transform raw recipe Documents into hierarchical chunks.

    Steps:
      1. Remove standard footer.
      2. Split by H1/H2 using MarkdownHeaderTextSplitter.
      3. Extract metadata (dish_name, category, difficulty, calories).
      4. Optionally split H3 subsections within ## 操作.
    """

    def __init__(self, remove_footer: bool = REMOVE_FOOTER, split_h3: bool = SPLIT_H3):
        self.remove_footer = remove_footer
        self.split_h3 = split_h3
        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )

    def transform_documents(self, documents: List[Document]) -> List[Document]:
        chunks: List[Document] = []
        for doc in documents:
            chunks.extend(self._process_one(doc))
        return chunks

    def _process_one(self, doc: Document) -> List[Document]:
        source = doc.metadata.get("source", "")
        category = doc.metadata.get("category", "unknown")
        content = doc.page_content

        if self.remove_footer:
            content = FOOTER_RE.sub("", content).rstrip()

        md_docs = self._md_splitter.split_text(content)
        if not md_docs:
            return []

        dish_name = self._extract_dish_name(md_docs)
        result: List[Document] = []

        for md_doc in md_docs:
            headers = md_doc.metadata
            text = md_doc.page_content.strip()
            if not text:
                continue

            if "H2" not in headers:
                result.append(Document(
                    page_content=text,
                    metadata={
                        "dish_name": dish_name,
                        "category": category,
                        "level": "dish",
                        "path": os.path.relpath(source, DISHES_DIR),
                        "difficulty": self._extract_difficulty(text),
                        "calories": self._extract_calories(text),
                    },
                ))
            elif self.split_h3 and headers.get("H2") == "操作":
                result.extend(self._split_h3_subsections(
                    text, dish_name, category, source
                ))
            else:
                section_type = headers.get("H2", "")
                result.append(Document(
                    page_content=text,
                    metadata={
                        "dish_name": dish_name,
                        "category": category,
                        "level": "section",
                        "section_type": section_type,
                        "path": os.path.relpath(source, DISHES_DIR),
                    },
                ))

        return result

    @staticmethod
    def _extract_dish_name(md_docs: List[Document]) -> str:
        for d in md_docs:
            h1 = d.metadata.get("H1", "")
            if h1:
                return re.sub(r"的做法$", "", h1).strip()
        return "unknown"

    @staticmethod
    def _extract_difficulty(text: str) -> str:
        m = re.search(r"预估烹饪难度[：:]\s*(\S+)", text)
        return m.group(1) if m else ""

    @staticmethod
    def _extract_calories(text: str) -> str:
        m = re.search(r"预估卡路里[：:]\s*(\S+)", text)
        return m.group(1) if m else ""

    def _split_h3_subsections(
        self, text: str, dish_name: str, category: str, source: str
    ) -> List[Document]:
        h3_re = re.compile(r"^###\s+(.*)", re.MULTILINE)
        positions = [(m.start(), m.group(1).strip()) for m in h3_re.finditer(text)]

        if not positions:
            return [Document(
                page_content=text,
                metadata={
                    "dish_name": dish_name, "category": category,
                    "level": "section", "section_type": "操作",
                    "path": os.path.relpath(source, DISHES_DIR),
                },
            )]

        chunks = []
        for i, (pos, heading) in enumerate(positions):
            end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            sub_text = text[pos:end].strip()
            if sub_text:
                chunks.append(Document(
                    page_content=sub_text,
                    metadata={
                        "dish_name": dish_name, "category": category,
                        "level": "subsection", "section_type": "操作",
                        "subsection_name": heading,
                        "path": os.path.relpath(source, DISHES_DIR),
                    },
                ))
        return chunks
