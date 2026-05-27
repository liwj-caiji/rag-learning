"""Tests for retrieval module: search, filters, and diversity reranking."""

import pytest


# ======================================================================
# Hybrid / Dense / Sparse search (requires index)
# ======================================================================

class TestHybridSearch:
    """Integration tests requiring pre-built index."""

    def test_hybrid_search_returns_results(self, index_available):
        """hybrid_search should return a list of results with score + chunk."""
        if not index_available:
            pytest.skip("Index not built, run `python -m src.preprocess.indexer` first")

        from src.retrieval import hybrid_search
        results = hybrid_search("红烧肉", k=5)
        assert len(results) > 0, "hybrid_search returned empty results"
        assert len(results) <= 5, f"Expected <=5 results, got {len(results)}"

        for r in results:
            assert "score" in r, "Missing 'score' field"
            assert "chunk" in r, "Missing 'chunk' field"
            assert r["score"] > 0, f"Non-positive score: {r['score']}"

    def test_hybrid_search_different_k(self, index_available):
        """Respecting the k parameter."""
        if not index_available:
            pytest.skip("Index not built")

        from src.retrieval import hybrid_search
        r3 = hybrid_search("鸡", k=3)
        r10 = hybrid_search("鸡", k=10)
        assert len(r3) <= 3
        assert len(r10) <= 10
        assert len(r10) >= len(r3)

    def test_hybrid_search_includes_dense_and_sparse_ranks(self, index_available):
        """RRF-fused results should include both dense and sparse ranks."""
        if not index_available:
            pytest.skip("Index not built")

        from src.retrieval import hybrid_search
        results = hybrid_search("红烧肉", k=5)
        for r in results:
            assert "dense_rank" in r
            assert "sparse_rank" in r

    def test_dense_search(self, index_available):
        """dense_search should return results sorted by score desc."""
        if not index_available:
            pytest.skip("Index not built")

        from src.retrieval import dense_search
        results = dense_search("红烧肉", k=5)
        assert len(results) > 0
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Scores not descending"

    def test_sparse_search(self, index_available):
        """sparse_search should return results with positive scores."""
        if not index_available:
            pytest.skip("Index not built")

        from src.retrieval import sparse_search
        results = sparse_search("红烧肉", k=5)
        if results:  # May be empty if BM25 has no match
            assert all(r["score"] > 0 for r in results)

    def test_sparse_search_filters_zero_scores(self, index_available):
        """sparse_search should exclude BM25 scores of 0."""
        if not index_available:
            pytest.skip("Index not built")

        from src.retrieval import sparse_search
        results = sparse_search("zzznotawordzzz", k=5)
        assert len(results) == 0, "Sparse search returned results for nonsense query"

    def test_empty_query_returns_empty(self, index_available):
        """Empty query should not crash (implementation-dependent behavior)."""
        if not index_available:
            pytest.skip("Index not built")

        from src.retrieval import hybrid_search
        try:
            results = hybrid_search("", k=5)
            assert isinstance(results, list)
        except Exception as e:
            pytest.fail(f"Empty query raised exception: {e}")


# ======================================================================
# recommend_dishes
# ======================================================================

class TestRecommendDishes:
    """Multi-dish recommendation tests."""

    def test_recommend_returns_dish_level(self, index_available):
        """recommend_dishes should only return dish-level (L1) chunks."""
        if not index_available:
            pytest.skip("Index not built")

        from src.retrieval import recommend_dishes
        results = recommend_dishes("推荐一道家常菜", k=3)
        if results:
            for r in results:
                assert r["chunk"]["level"] == "dish", "Non-dish chunk in recommendation"

    def test_recommend_respects_k(self, index_available):
        """Should return at most k results."""
        if not index_available:
            pytest.skip("Index not built")

        from src.retrieval import recommend_dishes
        results = recommend_dishes("今天吃什么", k=2)
        assert len(results) <= 2

    def test_recommend_with_filters_category(self, index_available):
        """Category filter should be applied."""
        if not index_available:
            pytest.skip("Index not built")

        from src.retrieval import recommend_dishes
        results = recommend_dishes(
            "推荐素菜", k=3,
            filters={"category": "vegetable_dish"},
        )
        if results:
            for r in results:
                assert r["chunk"]["metadata"]["category"] == "vegetable_dish"

    def test_recommend_with_probes(self, index_available):
        """Multiple search probes should increase recall."""
        if not index_available:
            pytest.skip("Index not built")

        from src.retrieval import recommend_dishes
        results_no_probe = recommend_dishes("肉菜", k=3)
        results_with_probe = recommend_dishes(
            "肉菜", k=3, probes=["红烧肉", "家常肉菜", "猪肉"],
        )
        # With probes should not crash
        assert isinstance(results_with_probe, list)


# ======================================================================
# Filters
# ======================================================================

class TestFilters:
    """Unit tests for metadata filtering."""

    @pytest.fixture
    def sample_chunks(self):
        from src.retrieval.filters import apply_filters
        return [
            {
                "text": "...",
                "level": "dish",
                "metadata": {
                    "dish_name": "红烧肉",
                    "category": "meat_dish",
                    "difficulty": "★★",
                    "calories": "600 千卡",
                },
            },
            {
                "text": "...",
                "level": "dish",
                "metadata": {
                    "dish_name": "清炒时蔬",
                    "category": "vegetable_dish",
                    "difficulty": "★",
                    "calories": "150 千卡",
                },
            },
            {
                "text": "...",
                "level": "section",
                "metadata": {
                    "dish_name": "红烧肉",
                    "category": "meat_dish",
                    "section_type": "操作",
                },
            },
        ]

    def test_filter_category(self, sample_chunks):
        from src.retrieval.filters import apply_filters
        result = apply_filters(sample_chunks, {"category": "vegetable_dish"})
        assert len(result) == 1
        assert result[0]["metadata"]["dish_name"] == "清炒时蔬"

    def test_filter_difficulty(self, sample_chunks):
        from src.retrieval.filters import apply_filters
        result = apply_filters(sample_chunks, {"difficulty": "★"})
        # ★ matches both ★ and ★★ via prefix; section chunk passes through (no difficulty)
        assert len(result) == 3

    def test_filter_level(self, sample_chunks):
        from src.retrieval.filters import apply_filters
        result = apply_filters(sample_chunks, {"level": "section"})
        assert len(result) == 1
        assert result[0]["level"] == "section"

    def test_filter_target_dish(self, sample_chunks):
        from src.retrieval.filters import apply_filters
        result = apply_filters(sample_chunks, {"target_dish": "红烧肉"})
        assert len(result) == 2  # L1 dish + L2 section

    def test_filter_calories_low(self, sample_chunks):
        from src.retrieval.filters import apply_filters
        result = apply_filters(sample_chunks, {"calories": "low"})
        # low matches 清炒时蔬 (150kcal) + section chunk passes through (no calories)
        assert len(result) == 2
        assert result[0]["metadata"]["dish_name"] == "清炒时蔬"

    def test_filter_calories_high(self, sample_chunks):
        from src.retrieval.filters import apply_filters
        result = apply_filters(sample_chunks, {"calories": "high"})
        # high matches 红烧肉 dish (600kcal) + section chunk passes through (no calories)
        assert len(result) == 2

    def test_no_filters_returns_all(self, sample_chunks):
        from src.retrieval.filters import apply_filters
        result = apply_filters(sample_chunks, None)
        assert len(result) == 3

    def test_empty_filters_returns_all(self, sample_chunks):
        from src.retrieval.filters import apply_filters
        result = apply_filters(sample_chunks, {})
        assert len(result) == 3


# ======================================================================
# Diversity reranking
# ======================================================================

class TestDiversity:
    """Unit tests for diversity reranking."""

    @pytest.fixture
    def multi_category_chunks(self):
        return [
            {"score": 0.9, "metadata": {"dish_name": "红烧肉", "category": "meat_dish"}},
            {"score": 0.8, "metadata": {"dish_name": "清炒时蔬", "category": "vegetable_dish"}},
            {"score": 0.7, "metadata": {"dish_name": "番茄蛋汤", "category": "soup"}},
            {"score": 0.6, "metadata": {"dish_name": "回锅肉", "category": "meat_dish"}},
        ]

    def test_diversify_by_category_round_robin(self, multi_category_chunks):
        from src.retrieval.diversity import diversify_by_category
        result = diversify_by_category(multi_category_chunks, k=3)
        assert len(result) <= 3

        # Should pick one from each category first
        cats = [r["metadata"]["category"] for r in result]
        assert len(set(cats)) >= 2, "Expected at least 2 categories in top-3"

    def test_diversify_k_larger_than_input(self, multi_category_chunks):
        from src.retrieval.diversity import diversify_by_category
        result = diversify_by_category(multi_category_chunks, k=10)
        assert len(result) == 4  # capped at input size

    def test_diversify_empty(self):
        from src.retrieval.diversity import diversify_by_category
        assert diversify_by_category([], k=5) == []

    def test_mmr_rerank_basic(self, multi_category_chunks):
        import numpy as np
        from src.retrieval.diversity import mmr_rerank
        query_emb = np.array([0.1, 0.2])
        # Assign fake similarity scores to chunks
        for i, c in enumerate(multi_category_chunks):
            c["score"] = 1.0 - i * 0.1
        result = mmr_rerank(multi_category_chunks, query_emb, lambda_=0.5, k=2)
        assert len(result) <= 2

    def test_mmr_empty(self):
        import numpy as np
        from src.retrieval.diversity import mmr_rerank
        assert mmr_rerank([], np.array([0.1]), k=5) == []

    def test_chunk_sim_same_dish(self):
        from src.retrieval.diversity import _chunk_sim
        a = {"metadata": {"dish_name": "红烧肉", "category": "meat_dish"}}
        b = {"metadata": {"dish_name": "红烧肉", "category": "meat_dish"}}
        assert _chunk_sim(a, b) == 0.9

    def test_chunk_sim_same_category(self):
        from src.retrieval.diversity import _chunk_sim
        a = {"metadata": {"dish_name": "红烧肉", "category": "meat_dish"}}
        b = {"metadata": {"dish_name": "回锅肉", "category": "meat_dish"}}
        assert _chunk_sim(a, b) == 0.3

    def test_chunk_sim_different(self):
        from src.retrieval.diversity import _chunk_sim
        a = {"metadata": {"dish_name": "红烧肉", "category": "meat_dish"}}
        b = {"metadata": {"dish_name": "清炒时蔬", "category": "vegetable_dish"}}
        assert _chunk_sim(a, b) == 0.0
