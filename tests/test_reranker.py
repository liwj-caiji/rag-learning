"""Tests for Cross-Encoder reranker (LangChain backend)."""

import pytest


class TestCrossEncoderReranker:
    """Unit tests for LangChain CrossEncoderReranker."""

    @pytest.fixture
    def sample_candidates(self):
        from langchain_core.documents import Document
        return [
            Document(
                page_content="麻婆豆腐是一道经典的川菜，主要原料包括豆腐、牛肉末、豆瓣酱、花椒粉。",
                metadata={"dish_name": "麻婆豆腐", "category": "meat_dish", "rrf_score": 0.032},
            ),
            Document(
                page_content="红烧肉是一道著名的上海菜，以五花肉为主要原料，配以酱油、糖、料酒等调料。",
                metadata={"dish_name": "红烧肉", "category": "meat_dish", "rrf_score": 0.028},
            ),
            Document(
                page_content="清炒时蔬是一道简单的素菜，需要新鲜蔬菜和蒜末。",
                metadata={"dish_name": "清炒时蔬", "category": "vegetable_dish", "rrf_score": 0.025},
            ),
        ]

    def test_reranker_returns_results(self, sample_candidates):
        """Reranker should return top_k results with ce_score in metadata."""
        from src_langchain.retrieval.reranker import get_reranker
        reranker = get_reranker()

        results = reranker.rerank("麻婆豆腐怎么做", sample_candidates, top_k=2)
        assert len(results) == 2
        for r in results:
            assert "rrf_score" in r.metadata
            assert "ce_score" in r.metadata

    def test_reranker_sorts_by_ce_score(self, sample_candidates):
        """Results should be sorted by CE score descending."""
        from src_langchain.retrieval.reranker import get_reranker
        reranker = get_reranker()

        results = reranker.rerank("麻婆豆腐怎么做", sample_candidates, top_k=3)
        scores = [r.metadata["ce_score"] for r in results]
        assert scores == sorted(scores, reverse=True), f"Scores not descending: {scores}"

    def test_reranker_first_result_most_relevant(self, sample_candidates):
        """Most relevant candidate should rank first."""
        from src_langchain.retrieval.reranker import get_reranker
        reranker = get_reranker()

        results = reranker.rerank("麻婆豆腐怎么做", sample_candidates, top_k=3)
        assert "麻婆豆腐" in results[0].metadata["dish_name"]

    def test_reranker_empty_candidates(self):
        """Empty candidate list should return empty."""
        from src_langchain.retrieval.reranker import get_reranker
        reranker = get_reranker()

        results = reranker.rerank("test", [], top_k=5)
        assert results == []

    def test_reranker_top_k_larger_than_candidates(self, sample_candidates):
        """top_k > len(candidates) should return all candidates."""
        from src_langchain.retrieval.reranker import get_reranker
        reranker = get_reranker()

        results = reranker.rerank("test", sample_candidates, top_k=10)
        assert len(results) == len(sample_candidates)

    def test_reranker_singleton(self):
        """get_reranker() should return the same instance."""
        from src_langchain.retrieval.reranker import get_reranker
        r1 = get_reranker()
        r2 = get_reranker()
        assert r1 is r2
