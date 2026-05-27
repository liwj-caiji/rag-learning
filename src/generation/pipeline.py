"""Full RAG pipeline: rewrite → retrieve → assemble → generate."""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

from src.config import (
    LLM_GEN_MODEL, LLM_GEN_API_BASE, LLM_API_KEY_ENV,
    PIPELINE_HOWTO_K, PIPELINE_HOWTO_DENSE_K, PIPELINE_HOWTO_SPARSE_K,
    PIPELINE_INGREDIENT_K, PIPELINE_INGREDIENT_DENSE_K, PIPELINE_INGREDIENT_SPARSE_K,
)
from ..preprocess.config import DISHES_DIR
from ..rewriting import IntentResult, QueryRewriter, LLMIntentClassifier
from ..retrieval import hybrid_search, recommend_dishes
from .base import Generator
from .template import TemplateGenerator

log = logging.getLogger("pipeline")


class RAGPipeline:
    """End-to-end RAG pipeline.

    Usage:
        pipeline = RAGPipeline()
        answer = pipeline.run("今天吃什么")
        print(answer)

    With LLM:
        pipeline = RAGPipeline(use_llm=True)
        answer = pipeline.run("麻婆豆腐怎么做")
    """

    def __init__(
        self,
        rewriter: Optional[QueryRewriter] = None,
        generator: Optional[Generator] = None,
        use_llm: bool = False,
        llm_model: str = LLM_GEN_MODEL,
        llm_api_base: str = LLM_GEN_API_BASE,
    ):
        if use_llm and (not rewriter or not generator):
            api_key = os.environ.get(LLM_API_KEY_ENV)
            if not api_key:
                raise ValueError(
                    "DEEPSEEK_API_KEY environment variable is required "
                    "when use_llm=True"
                )

        self.rewriter = rewriter or (
            LLMIntentClassifier(model=llm_model, api_base=llm_api_base)
            if use_llm
            else QueryRewriter()
        )
        self.generator = generator or (
            self._make_llm_generator(llm_model, llm_api_base)
            if use_llm
            else TemplateGenerator()
        )
        self._llm_model = llm_model
        self._llm_api_base = llm_api_base

    @staticmethod
    def _make_llm_generator(model: str, api_base: str) -> Generator:
        from .llm_generator import LLMGenerator
        return LLMGenerator(model=model, api_base=api_base)

    @staticmethod
    def _enrich_with_full_recipe(
        results: List[Dict],
        dish_query: str,
    ) -> Optional[List[Dict]]:
        """Load full source file(s) and prepend complete recipe content.

        When the target dish is identified, this method reads the original
        markdown file so the LLM receives the full recipe (description +
        ingredients + steps) rather than isolated chunk snippets.
        """
        if not dish_query:
            return None

        # Collect source file paths from chunks matching the target dish
        source_paths = set()
        for item in results:
            chunk = item.get("chunk", item)
            meta = chunk.get("metadata", {})
            if dish_query in meta.get("dish_name", ""):
                path = meta.get("path", "")
                if path:
                    source_paths.add(path)

        if not source_paths:
            return None

        enriched = list(results)  # keep original chunks as well

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

            # Derive dish name from path
            dish_name = os.path.splitext(os.path.basename(rel_path))[0]

            enriched.insert(0, {
                "chunk": {
                    "text": content,
                    "level": "full_recipe",
                    "metadata": {
                        "dish_name": dish_name,
                        "path": rel_path,
                        "section_type": "完整食谱",
                    },
                },
            })
            log.info("Enriched context with full recipe: %s (len=%d)",
                     rel_path, len(content))

        return enriched

    def run(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        """Execute full RAG pipeline."""
        log.info("Query: %r | top_k=%d", query, top_k)

        t0 = time.time()
        intent_result = self.rewriter.rewrite(query)
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

        log.info("Total: %.2fs", time.time() - t0)
        return answer

    def _retrieve(
        self,
        intent_result: IntentResult,
        top_k: int,
        query: str = "",
    ) -> List[Dict]:
        """Select retrieval strategy based on intent."""
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
            # Prefer 操作 sections matching the target dish name
            op_results = [
                r for r in results
                if r["chunk"]["metadata"].get("section_type") == "操作"
                and (
                    not dish_query
                    or dish_query in r["chunk"]["metadata"].get("dish_name", "")
                )
            ]
            if op_results:
                chosen = op_results[:top_k]
            else:
                # Fallback: dish-level chunks for the target dish
                dish_results = [
                    r for r in results
                    if dish_query and dish_query in r["chunk"]["metadata"].get("dish_name", "")
                ]
                chosen = dish_results[:top_k] if dish_results else results[:top_k]
            # Enrich with full recipe file for complete context
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
            ing_results = [
                r for r in results
                if r["chunk"]["metadata"].get("section_type") == "必备原料和工具"
                and (
                    not dish_query
                    or dish_query in r["chunk"]["metadata"].get("dish_name", "")
                )
            ]
            if ing_results:
                chosen = ing_results[:top_k]
            else:
                dish_results = [
                    r for r in results
                    if dish_query and dish_query in r["chunk"]["metadata"].get("dish_name", "")
                ]
                chosen = dish_results[:top_k] if dish_results else results[:top_k]
            # Enrich with full recipe file for complete context
            enriched = self._enrich_with_full_recipe(chosen, dish_query)
            return enriched if enriched else chosen

        # Fallback: generic hybrid search
        return hybrid_search(query, k=top_k)

    def trace(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict:
        """Run pipeline and return detailed trace for debugging."""
        intent_result = self.rewriter.rewrite(query)
        context = self._retrieve(intent_result, top_k, query)
        answer = self.generator.generate(
            query, context, intent_result.intent,
            target_dish=intent_result.target_dish,
        )

        return {
            "query": query,
            "intent": intent_result.intent,
            "rewritten": intent_result.rewritten,
            "probes": intent_result.probes,
            "filters": intent_result.filters,
            "target_dish": intent_result.target_dish,
            "num_chunks": len(context),
            "chunks": [
                {
                    "dish": c["chunk"]["metadata"].get("dish_name") or "",
                    "level": c["chunk"]["level"] or "",
                    "section": c["chunk"]["metadata"].get("section_type") or "",
                    "category": c["chunk"]["metadata"].get("category") or "",
                    "text": c["chunk"].get("text", ""),
                }
                for c in context
            ],
            "answer": answer,
        }
