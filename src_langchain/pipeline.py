"""Full RAG pipeline orchestrated with LangChain components.

Orchestrates: rewrite -> retrieve -> enrich -> generate.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

from langchain_core.documents import Document

from .config import (
    LLM_GEN_MODEL, LLM_GEN_API_BASE, LLM_API_KEY_ENV,
    DISHES_DIR,
    PIPELINE_HOWTO_K, PIPELINE_HOWTO_DENSE_K, PIPELINE_HOWTO_SPARSE_K,
    PIPELINE_INGREDIENT_K, PIPELINE_INGREDIENT_DENSE_K, PIPELINE_INGREDIENT_SPARSE_K,
)
from .rewriting import IntentResult, get_intent_classifier
from .retrieval import hybrid_search, recommend_dishes
from .generation import TemplateGenerator, LLMGenerator, Generator
from .tracing import get_langfuse_handler

log = logging.getLogger("pipeline.lc")

try:
    from langfuse import observe
except ImportError:
    def observe(**kwargs):  # type: ignore[no-redef]
        """No-op decorator when langfuse is not installed."""
        return lambda fn: fn


class RAGPipeline:
    """End-to-end RAG pipeline using LangChain-managed components.

    Usage:
        pipeline = RAGPipeline()
        answer = pipeline.run("今天吃什么")

        pipeline = RAGPipeline(use_llm=True)
        answer = pipeline.run("麻婆豆腐怎么做")
    """

    def __init__(
        self,
        use_llm: bool = False,
        llm_model: str = LLM_GEN_MODEL,
        llm_api_base: str = LLM_GEN_API_BASE,
    ):
        if use_llm:
            api_key = os.environ.get(LLM_API_KEY_ENV)
            if not api_key:
                raise ValueError(
                    f"{LLM_API_KEY_ENV} environment variable is required "
                    f"when use_llm=True"
                )

        self._llm_model = llm_model
        self._llm_api_base = llm_api_base
        self._use_llm = use_llm

        self._langfuse_handler = get_langfuse_handler()
        _callbacks = [self._langfuse_handler] if self._langfuse_handler else None

        self.rewriter = get_intent_classifier(
            use_llm=use_llm,
            model=llm_model,
            api_base=llm_api_base,
            callbacks=_callbacks,
        )
        self.generator: Generator = (
            LLMGenerator(model=llm_model, api_base=llm_api_base, callbacks=_callbacks)
            if use_llm
            else TemplateGenerator()
        )

        self._lf_client = None
        if self._langfuse_handler:
            try:
                from langfuse import Langfuse
                self._lf_client = Langfuse()
            except Exception:
                pass

    @observe(name="RAGPipeline.run")
    def run(self, query: str, top_k: int = 5) -> str:
        """Execute full RAG pipeline and return answer string."""
        log.info("Query: %r | top_k=%d", query, top_k)

        t0 = time.time()
        intent_result = self.rewriter.classify(query)
        log.info("Rewrite: intent=%s target=%s | %.2fs",
                 intent_result.intent, intent_result.target_dish, time.time() - t0)

        t1 = time.time()
        context = self._retrieve(intent_result, top_k, query)
        log.info("Retrieve: %d chunks | %.2fs", len(context), time.time() - t1)

        t2 = time.time()
        answer = self.generator.generate(
            query, context, intent_result.intent,
            target_dish=intent_result.target_dish,
        )
        log.info("Generate: %d chars | %.2fs | target=%s",
                 len(answer), time.time() - t2, intent_result.target_dish)

        total_elapsed = time.time() - t0
        log.info("Total: %.2fs", total_elapsed)

        self._enrich_trace(query, intent_result, context, answer, total_elapsed)
        return answer

    @observe(name="RAGPipeline.trace")
    def trace(self, query: str, top_k: int = 5) -> Dict:
        """Run pipeline and return detailed trace for debugging / UI."""
        t0 = time.time()
        intent_result = self.rewriter.classify(query)
        context = self._retrieve(intent_result, top_k, query)
        answer = self.generator.generate(
            query, context, intent_result.intent,
            target_dish=intent_result.target_dish,
        )

        result = {
            "query": query,
            "intent": intent_result.intent,
            "rewritten": intent_result.rewritten,
            "probes": intent_result.probes,
            "filters": intent_result.filters,
            "target_dish": intent_result.target_dish,
            "num_chunks": len(context),
            "chunks": [
                {
                    "dish": doc.metadata.get("dish_name") or "",
                    "level": doc.metadata.get("level") or "",
                    "section": doc.metadata.get("section_type") or "",
                    "category": doc.metadata.get("category") or "",
                    "text": doc.page_content,
                }
                for doc in context
            ],
            "answer": answer,
        }

        total_elapsed = time.time() - t0
        self._enrich_trace(query, intent_result, context, answer, total_elapsed)
        return result

    # ------------------------------------------------------------------
    # Retrieval routing
    # ------------------------------------------------------------------

    def _retrieve(
        self,
        intent_result: IntentResult,
        top_k: int,
        query: str = "",
    ) -> List[Document]:
        intent = intent_result.intent

        if intent == "recommendation":
            return recommend_dishes(
                query=intent_result.rewritten or intent_result.target_dish or "",
                k=top_k,
                filters=intent_result.filters,
                diversify=True,
                probes=intent_result.probes,
            )

        if intent == "howto":
            dish_query = intent_result.target_dish or ""
            query_str = f"{dish_query} 操作" if dish_query else intent_result.rewritten
            results = hybrid_search(
                query_str, k=PIPELINE_HOWTO_K,
                dense_k=PIPELINE_HOWTO_DENSE_K, sparse_k=PIPELINE_HOWTO_SPARSE_K,
            )
            chosen = self._filter_by_section(results, dish_query, "操作", top_k)
            enriched = self._enrich_with_full_recipe(chosen, dish_query)
            return enriched if enriched else chosen

        if intent == "ingredient":
            dish_query = intent_result.target_dish or ""
            query_str = (
                f"{dish_query} 必备原料和工具" if dish_query else intent_result.rewritten
            )
            results = hybrid_search(
                query_str, k=PIPELINE_INGREDIENT_K,
                dense_k=PIPELINE_INGREDIENT_DENSE_K, sparse_k=PIPELINE_INGREDIENT_SPARSE_K,
            )
            chosen = self._filter_by_section(results, dish_query, "必备原料和工具", top_k)
            enriched = self._enrich_with_full_recipe(chosen, dish_query)
            return enriched if enriched else chosen

        return hybrid_search(query, k=top_k)

    @staticmethod
    def _filter_by_section(
        results: List[Document],
        dish_query: str,
        section_type: str,
        top_k: int,
    ) -> List[Document]:
        section_results = [
            doc for doc in results
            if doc.metadata.get("section_type") == section_type
            and (not dish_query
                 or dish_query in doc.metadata.get("dish_name", ""))
        ]
        if section_results:
            return section_results[:top_k]

        dish_results = [
            doc for doc in results
            if dish_query and dish_query in doc.metadata.get("dish_name", "")
        ]
        return dish_results[:top_k] if dish_results else results[:top_k]

    @staticmethod
    def _enrich_with_full_recipe(
        results: List[Document],
        dish_query: str,
    ) -> Optional[List[Document]]:
        if not dish_query:
            return None

        source_paths = set()
        for doc in results:
            meta = doc.metadata
            if dish_query in meta.get("dish_name", ""):
                path = meta.get("path", "")
                if path:
                    source_paths.add(path)

        if not source_paths:
            return None

        enriched = list(results)

        for rel_path in source_paths:
            full_path = os.path.join(DISHES_DIR, rel_path)
            if not os.path.exists(full_path):
                log.warning("Source file not found: %s", full_path)
                continue

            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                log.warning("Failed to read source file %s: %s", full_path, e)
                continue

            dish_name = os.path.splitext(os.path.basename(rel_path))[0]

            enriched.insert(0, Document(
                page_content=content,
                metadata={
                    "dish_name": dish_name,
                    "level": "full_recipe",
                    "path": rel_path,
                    "section_type": "完整食谱",
                },
            ))
            log.info("Enriched context with full recipe: %s (len=%d)",
                     rel_path, len(content))

        return enriched

    # ------------------------------------------------------------------
    # Langfuse trace enrichment (for auto-scoring)
    # ------------------------------------------------------------------

    def _enrich_trace(
        self,
        query: str,
        intent_result: IntentResult,
        context: List[Document],
        answer: str,
        total_elapsed: float,
    ) -> None:
        """Attach metadata to the current Langfuse span (trace).

        In langfuse v4.x, @observe creates a span whose metadata flows
        through to the trace. Tags are not supported via the span API,
        so intent/model/backend info is included in metadata instead.
        """
        if not self._lf_client:
            return

        context_texts = [doc.page_content for doc in context]

        try:
            self._lf_client.update_current_span(
                name=f"RAG: {query[:40]}{'...' if len(query) > 40 else ''}",
                metadata={
                    "intent": intent_result.intent,
                    "target_dish": intent_result.target_dish or "",
                    "rewritten_query": intent_result.rewritten or "",
                    "num_chunks": len(context),
                    "total_elapsed_s": round(total_elapsed, 3),
                    "model": self._llm_model,
                    "use_llm": self._use_llm,
                    "filters": intent_result.filters or {},
                    "probes": intent_result.probes or [],
                    "contexts": context_texts,
                    "tag_intent": intent_result.intent,
                    "tag_model": self._llm_model,
                    "tag_backend": "langchain",
                    "tag_mode": "llm" if self._use_llm else "template",
                },
            )
        except Exception:
            log.debug("Failed to enrich Langfuse span", exc_info=True)
