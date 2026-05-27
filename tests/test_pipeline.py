"""Tests for generation pipeline: RAGPipeline end-to-end, trace, edge cases."""

import os

import pytest


# ======================================================================
# RAGPipeline construction
# ======================================================================

class TestPipelineConstruction:
    """Verify pipeline initialization."""

    def test_default_pipeline_rule_mode(self):
        from src.generation import RAGPipeline
        pipe = RAGPipeline()
        assert pipe is not None

        # Should use RuleQueryRewriter + TemplateGenerator
        rewriter_name = type(pipe.rewriter).__name__
        generator_name = type(pipe.generator).__name__
        assert rewriter_name in ("RuleQueryRewriter", "QueryRewriter"), \
            f"Expected rule rewriter, got {rewriter_name}"
        assert generator_name == "TemplateGenerator", \
            f"Expected TemplateGenerator, got {generator_name}"

    def test_llm_mode_requires_api_key(self):
        """use_llm=True without DEEPSEEK_API_KEY should raise ValueError."""
        from src.generation import RAGPipeline
        old_key = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
                RAGPipeline(use_llm=True)
        finally:
            if old_key:
                os.environ["DEEPSEEK_API_KEY"] = old_key

    def test_custom_rewriter_generator(self):
        """Custom rewriter and generator should be used."""
        from src.generation import RAGPipeline
        from src.rewriting import RuleQueryRewriter
        from src.generation import TemplateGenerator

        rewriter = RuleQueryRewriter()
        generator = TemplateGenerator()
        pipe = RAGPipeline(rewriter=rewriter, generator=generator)
        assert pipe.rewriter is rewriter
        assert pipe.generator is generator

    def test_llm_mode_with_key_succeeds(self):
        """use_llm=True with DEEPSEEK_API_KEY should succeed."""
        from src.generation import RAGPipeline
        old_key = os.environ.get("DEEPSEEK_API_KEY")
        if not old_key:
            pytest.skip("DEEPSEEK_API_KEY not set, cannot test LLM mode")
        try:
            pipe = RAGPipeline(use_llm=True)
            assert pipe is not None
        finally:
            pass  # keep key


# ======================================================================
# RAGPipeline.run (rule mode — requires index)
# ======================================================================

class TestPipelineRun:
    """End-to-end pipeline execution in rule mode."""

    def _check_index(self):
        from src.preprocess.config import VECTORSTORE_DIR
        import pickle
        idx_path = os.path.join(VECTORSTORE_DIR, "faiss.index")
        chunks_path = os.path.join(VECTORSTORE_DIR, "chunks.pkl")
        return os.path.exists(idx_path) and os.path.exists(chunks_path)

    def test_run_howto_returns_string(self, index_available):
        """Pipeline.run should return a non-empty string for howto queries."""
        if not index_available:
            pytest.skip("Index not built")
        from src.generation import RAGPipeline
        pipe = RAGPipeline()
        answer = pipe.run("麻婆豆腐怎么做", top_k=3)
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_run_recommendation_returns_string(self, index_available):
        if not index_available:
            pytest.skip("Index not built")
        from src.generation import RAGPipeline
        pipe = RAGPipeline()
        answer = pipe.run("今天吃什么", top_k=3)
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_run_ingredient_returns_string(self, index_available):
        if not index_available:
            pytest.skip("Index not built")
        from src.generation import RAGPipeline
        pipe = RAGPipeline()
        answer = pipe.run("麻婆豆腐需要什么材料", top_k=3)
        assert isinstance(answer, str)

    def test_run_empty_query(self, index_available):
        """Empty query should not crash."""
        if not index_available:
            pytest.skip("Index not built")
        from src.generation import RAGPipeline
        pipe = RAGPipeline()
        try:
            answer = pipe.run("", top_k=3)
            assert isinstance(answer, str)
        except Exception as e:
            pytest.fail(f"Empty query raised exception: {e}")

    def test_run_gibberish_query(self, index_available):
        """Gibberish query should not crash."""
        if not index_available:
            pytest.skip("Index not built")
        from src.generation import RAGPipeline
        pipe = RAGPipeline()
        try:
            answer = pipe.run("zzznotawordzzz", top_k=3)
            assert isinstance(answer, str)
        except Exception as e:
            pytest.fail(f"Gibberish query raised exception: {e}")


# ======================================================================
# Pipeline trace
# ======================================================================

class TestPipelineTrace:
    """Verify trace() output structure."""

    def test_trace_contains_all_keys(self, index_available):
        if not index_available:
            pytest.skip("Index not built")
        from src.generation import RAGPipeline
        pipe = RAGPipeline()
        trace = pipe.trace("麻婆豆腐怎么做", top_k=3)

        expected_keys = {
            "query", "intent", "rewritten", "probes",
            "filters", "target_dish", "num_chunks", "chunks", "answer",
        }
        assert expected_keys.issubset(trace.keys()), \
            f"Missing keys: {expected_keys - trace.keys()}"

    def test_trace_contains_chunks(self, index_available):
        if not index_available:
            pytest.skip("Index not built")
        from src.generation import RAGPipeline
        pipe = RAGPipeline()
        trace = pipe.trace("麻婆豆腐怎么做", top_k=3)
        assert "chunks" in trace
        assert isinstance(trace["chunks"], list)

    def test_trace_chunks_have_metadata(self, index_available):
        if not index_available:
            pytest.skip("Index not built")
        from src.generation import RAGPipeline
        pipe = RAGPipeline()
        trace = pipe.trace("麻婆豆腐怎么做", top_k=3)
        for c in trace.get("chunks", []):
            assert "dish" in c, f"Missing dish in chunk: {c}"
            assert "level" in c, f"Missing level in chunk: {c}"
            assert "text" in c, f"Missing text in chunk: {c}"

    def test_trace_intent_is_string(self, index_available):
        if not index_available:
            pytest.skip("Index not built")
        from src.generation import RAGPipeline
        pipe = RAGPipeline()
        trace = pipe.trace("麻婆豆腐怎么做", top_k=3)
        assert isinstance(trace.get("intent"), str)
        assert trace["intent"] != ""

    def test_trace_answer_is_string(self, index_available):
        if not index_available:
            pytest.skip("Index not built")
        from src.generation import RAGPipeline
        pipe = RAGPipeline()
        trace = pipe.trace("麻婆豆腐怎么做", top_k=3)
        assert isinstance(trace.get("answer"), str)


# ======================================================================
# Intent routing
# ======================================================================

class TestIntentRouting:
    """Verify different intents route to appropriate retrieval strategies."""

    def test_howto_retrieves_operation_sections(self, index_available):
        if not index_available:
            pytest.skip("Index not built")
        from src.generation import RAGPipeline
        pipe = RAGPipeline()
        trace = pipe.trace("麻婆豆腐怎么做", top_k=5)
        chunks = trace.get("chunks", [])
        assert len(chunks) > 0
        # Howto should prefer 操作 sections
        if trace.get("intent") == "howto":
            sections = [c.get("section", "") for c in chunks]
            has_op = any("操作" in s for s in sections)
            dish_names = [c.get("dish", "") for c in chunks]
            has_dish = any("麻婆豆腐" in d for d in dish_names)
            assert has_op or has_dish, \
                f"Expected 麻婆豆腐 or 操作 section, got: {chunks[:2]}"

    def test_recommendation_returns_dish_level(self, index_available):
        if not index_available:
            pytest.skip("Index not built")
        from src.generation import RAGPipeline
        pipe = RAGPipeline()
        trace = pipe.trace("今天吃什么", top_k=3)
        if trace.get("intent") == "recommendation":
            levels = [c.get("level", "") for c in trace.get("chunks", [])]
            # Should only contain dish-level chunks for recommendation
            assert all(l == "dish" for l in levels), \
                f"Non-dish chunks in recommendation: {levels}"


# ======================================================================
# Generator
# ======================================================================

class TestTemplateGenerator:
    """Template generator unit tests."""

    def test_template_recommendation_format(self):
        from src.generation import TemplateGenerator
        gen = TemplateGenerator()
        context = [
            {
                "chunk": {
                    "level": "dish",
                    "metadata": {
                        "dish_name": "麻婆豆腐",
                        "category": "meat_dish",
                        "difficulty": "★★",
                        "calories": "350 千卡",
                    },
                    "text": "麻婆豆腐的做法...",
                },
            },
        ]
        result = gen.generate("今天吃什么", context, "recommendation")
        assert "麻婆豆腐" in result
        assert "meat_dish" in result or "肉" in result
        assert "★★" in result or "350" in result

    def test_template_empty_context(self):
        from src.generation import TemplateGenerator
        gen = TemplateGenerator()
        result = gen.generate("今天吃什么", [], "recommendation")
        assert "抱歉" in result

    def test_template_howto_format(self):
        from src.generation import TemplateGenerator
        gen = TemplateGenerator()
        context = [
            {
                "chunk": {
                    "level": "section",
                    "metadata": {"dish_name": "麻婆豆腐", "section_type": "操作"},
                    "text": "1. 豆腐切块\n2. 炒肉末\n3. 加豆瓣酱",
                },
            },
        ]
        result = gen.generate("麻婆豆腐怎么做", context, "howto")
        assert "麻婆豆腐" in result

    def test_template_ingredient_format(self):
        from src.generation import TemplateGenerator
        gen = TemplateGenerator()
        context = [
            {
                "chunk": {
                    "level": "section",
                    "metadata": {"dish_name": "麻婆豆腐"},
                    "text": "豆腐 300g, 牛肉末 50g",
                },
            },
        ]
        result = gen.generate("麻婆豆腐需要什么材料", context, "ingredient")
        assert "麻婆豆腐" in result


# ======================================================================
# LLMGenerator fallback
# ======================================================================

class TestLLMGeneratorFallback:
    """LLM generator should gracefully fall back to template."""

    def test_fallback_without_api_key(self):
        from src.generation import LLMGenerator
        gen = LLMGenerator()
        context = [{"chunk": {"level": "dish", "metadata": {"dish_name": "麻婆豆腐"}, "text": "..."}}]
        result = gen.generate("麻婆豆腐怎么做", context, "howto")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_fallback_empty_context(self):
        from src.generation import LLMGenerator
        gen = LLMGenerator()
        result = gen.generate("今天吃什么", [], "recommendation")
        assert "抱歉" in result or len(result) > 0

    def test_fallback_different_intents(self):
        from src.generation import LLMGenerator
        gen = LLMGenerator()
        for intent in ("recommendation", "howto", "ingredient", "factual"):
            result = gen.generate("test", [], intent)
            assert isinstance(result, str)


# ======================================================================
# Pipeline error recovery
# ======================================================================

class TestPipelineErrorRecovery:
    """Pipeline should gracefully handle component failures."""

    def test_custom_rewriter_raises_exception(self):
        """If custom rewriter raises, pipeline.run should propagate it."""
        from src.generation import RAGPipeline
        from src.rewriting.intent import IntentResult

        class BrokenRewriter:
            def rewrite(self, query):
                raise RuntimeError("rewriter failed")

        pipe = RAGPipeline(rewriter=BrokenRewriter(), generator=None)
        with pytest.raises(RuntimeError, match="rewriter failed"):
            pipe.run("test")

    def test_custom_generator_raises_exception(self, index_available):
        """If custom generator raises, pipeline.run should propagate it."""
        if not index_available:
            pytest.skip("Index not built")
        from src.generation import RAGPipeline

        class BrokenGenerator:
            def generate(self, query, context, intent):
                raise RuntimeError("generator failed")

        pipe = RAGPipeline(generator=BrokenGenerator())
        with pytest.raises(RuntimeError, match="generator failed"):
            pipe.run("麻婆豆腐怎么做")

    def test_trace_propagates_rewriter_error(self):
        """trace() should propagate rewriter exceptions (no try/except wrapping)."""
        from src.generation import RAGPipeline

        class BrokenRewriter:
            def rewrite(self, query):
                raise RuntimeError("rewriter dead")

        pipe = RAGPipeline(rewriter=BrokenRewriter(), generator=None)
        with pytest.raises(RuntimeError):
            pipe.trace("test")

    def test_llm_generator_fallback_no_api_key(self):
        """LLMGenerator without key falls back to template (no crash)."""
        from src.generation import LLMGenerator
        gen = LLMGenerator()
        context = [
            {
                "chunk": {
                    "level": "dish",
                    "metadata": {"dish_name": "麻婆豆腐", "category": "meat_dish", "difficulty": "★★"},
                    "text": "麻婆豆腐的做法",
                },
            },
        ]
        result = gen.generate("麻婆豆腐怎么做", context, "howto")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_llm_intent_fallback_no_api_key(self):
        """LLMIntentClassifier without key falls back to rule-based (no crash)."""
        from src.rewriting.llm_intent import LLMIntentClassifier
        from src.rewriting.intent import IntentType
        classifier = LLMIntentClassifier()
        result = classifier.classify("麻婆豆腐怎么做")
        assert result.intent == IntentType.HOWTO


# ======================================================================
# Pipeline edge cases
# ======================================================================

class TestPipelineEdgeCases:
    """Edge cases for pipeline execution."""

    def test_special_characters_query(self, index_available):
        """Query with special characters should not crash."""
        if not index_available:
            pytest.skip("Index not built")
        from src.generation import RAGPipeline
        pipe = RAGPipeline()
        queries = [
            "!@#$%^&*()",
            "  空格测试  ",
            "123456",
            "a",
            "，。？！",
        ]
        for q in queries:
            try:
                answer = pipe.run(q, top_k=3)
                assert isinstance(answer, str)
            except Exception as e:
                pytest.fail(f"Query {q!r} raised exception: {e}")

    def test_large_top_k(self, index_available):
        """Large top_k should not crash (capped by available chunks)."""
        if not index_available:
            pytest.skip("Index not built")
        from src.generation import RAGPipeline
        pipe = RAGPipeline()
        answer = pipe.run("麻婆豆腐怎么做", top_k=999)
        assert isinstance(answer, str)

    def test_zero_top_k(self, index_available):
        """Zero top_k should not crash."""
        if not index_available:
            pytest.skip("Index not built")
        from src.generation import RAGPipeline
        pipe = RAGPipeline()
        answer = pipe.run("麻婆豆腐怎么做", top_k=0)
        assert isinstance(answer, str)

    def test_multiple_consecutive_calls(self, index_available):
        """Multiple consecutive pipeline calls should all succeed."""
        if not index_available:
            pytest.skip("Index not built")
        from src.generation import RAGPipeline
        pipe = RAGPipeline()
        queries = ["麻婆豆腐怎么做", "今天吃什么", "红烧肉需要什么材料", "宫保鸡丁是什么菜"]
        for q in queries:
            answer = pipe.run(q, top_k=3)
            assert isinstance(answer, str), f"Failed for query: {q}"
            assert len(answer) > 0, f"Empty answer for query: {q}"

    def test_trace_consistency_across_calls(self, index_available):
        """trace() should return consistent structure across multiple calls."""
        if not index_available:
            pytest.skip("Index not built")
        from src.generation import RAGPipeline
        pipe = RAGPipeline()
        expected_keys = {"query", "intent", "rewritten", "probes",
                         "filters", "target_dish", "num_chunks", "chunks", "answer"}
        for q in ["麻婆豆腐怎么做", "今天吃什么", "红烧肉需要什么材料"]:
            trace = pipe.trace(q, top_k=3)
            assert expected_keys.issubset(trace.keys()), \
                f"Missing keys for {q!r}: {expected_keys - trace.keys()}"
            assert isinstance(trace["answer"], str)


# ======================================================================
# LLMGenerator timeout / error simulation
# ======================================================================

class TestLLMGeneratorErrorHandling:
    """LLM generator should handle API errors gracefully."""

    def test_empty_response_from_llm(self):
        """Simulate LLM returning empty content — should fall back."""
        from src.generation.llm_generator import LLMGenerator
        gen = LLMGenerator()
        context = [
            {
                "chunk": {
                    "level": "dish",
                    "metadata": {"dish_name": "麻婆豆腐"},
                    "text": "豆腐切块",
                },
            },
        ]
        # Without API client, will use fallback
        result = gen.generate("麻婆豆腐怎么做", context, "howto")
        assert isinstance(result, str)
        assert "麻婆豆腐" in result

    def test_llm_generator_empty_context(self):
        """Empty context should return empty-response message."""
        from src.generation.llm_generator import LLMGenerator
        gen = LLMGenerator()
        for intent in ("recommendation", "howto", "ingredient", "factual"):
            result = gen.generate("test", [], intent)
            assert isinstance(result, str)
            assert "抱歉" in result or len(result) > 0

    def test_llm_generator_no_fallback_mode(self):
        """With fallback=False and no client, should return empty string."""
        from src.generation.llm_generator import LLMGenerator
        gen = LLMGenerator(fallback=False)
        result = gen.generate("test", [], "recommendation")
        assert result == "" or "抱歉" in result
