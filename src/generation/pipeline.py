"""Full RAG pipeline: rewrite → retrieve → assemble → generate."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from ..rewriting import IntentResult, QueryRewriter, LLMIntentClassifier
from ..retrieval import hybrid_search, recommend_dishes
from .base import Generator
from .template import TemplateGenerator


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
        llm_model: str = "deepseek-v4-flash",
        llm_api_base: str = "https://api.deepseek.com",
    ):
        if use_llm and (not rewriter or not generator):
            api_key = os.environ.get("DEEPSEEK_API_KEY")
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

    def run(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        """Execute full RAG pipeline.

        1. Rewrite: classify intent, extract constraints, generate probes
        2. Retrieve: select strategy based on intent
        3. Assemble: build structured context
        4. Generate: produce final response
        """
        # 1. Rewrite
        intent_result = self.rewriter.rewrite(query)

        # 2. Retrieve
        context = self._retrieve(intent_result, top_k)

        # 3. Generate
        return self.generator.generate(query, context, intent_result.intent)

    def _retrieve(
        self,
        intent_result: IntentResult,
        top_k: int,
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
            results = hybrid_search(query_str, k=50, dense_k=100, sparse_k=100)
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
                return op_results[:top_k]
            # Fallback: dish-level chunks for the target dish
            dish_results = [
                r for r in results
                if dish_query and dish_query in r["chunk"]["metadata"].get("dish_name", "")
            ]
            return dish_results[:top_k] if dish_results else results[:top_k]

        if intent == "ingredient":
            dish_query = intent_result.target_dish or ""
            query_str = (
                f"{dish_query} 必备原料和工具" if dish_query else intent_result.rewritten
            )
            results = hybrid_search(query_str, k=50, dense_k=100, sparse_k=100)
            ing_results = [
                r for r in results
                if r["chunk"]["metadata"].get("section_type") == "必备原料和工具"
                and (
                    not dish_query
                    or dish_query in r["chunk"]["metadata"].get("dish_name", "")
                )
            ]
            if ing_results:
                return ing_results[:top_k]
            dish_results = [
                r for r in results
                if dish_query and dish_query in r["chunk"]["metadata"].get("dish_name", "")
            ]
            return dish_results[:top_k] if dish_results else results[:top_k]

        # Fallback: generic hybrid search
        return hybrid_search(query, k=top_k)

    def trace(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict:
        """Run pipeline and return detailed trace for debugging."""
        intent_result = self.rewriter.rewrite(query)
        context = self._retrieve(intent_result, top_k)
        answer = self.generator.generate(query, context, intent_result.intent)

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
                    "dish": c["chunk"]["metadata"].get("dish_name"),
                    "level": c["chunk"]["level"],
                    "section": c["chunk"]["metadata"].get("section_type"),
                }
                for c in context
            ],
            "answer": answer,
        }
