"""Tests for streaming generation (LangChain backend)."""

import os
import pytest


class TestStreamingGeneration:
    """Unit tests for streaming LLM generation."""

    @pytest.fixture
    def sample_context(self):
        from langchain_core.documents import Document
        return [
            Document(
                page_content="麻婆豆腐做法：1.豆腐切块 2.炒牛肉末 3.加豆瓣酱 4.加花椒粉 5.出锅。",
                metadata={"dish_name": "麻婆豆腐", "category": "meat_dish", "section_type": "操作"},
            ),
        ]

    def test_generate_stream_returns_tokens(self, sample_context):
        """generate_stream should yield string tokens."""
        if not os.environ.get("DEEPSEEK_API_KEY"):
            pytest.skip("DEEPSEEK_API_KEY not set")

        from src_langchain.generation.llm_chain import LLMGenerator
        generator = LLMGenerator()

        tokens = list(generator.generate_stream(
            "麻婆豆腐怎么做", sample_context, "howto", target_dish="麻婆豆腐",
        ))
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
        answer = "".join(tokens)
        assert len(answer) > 0

    def test_generate_stream_empty_context(self):
        """Empty context should return fallback response."""
        if not os.environ.get("DEEPSEEK_API_KEY"):
            pytest.skip("DEEPSEEK_API_KEY not set")

        from src_langchain.generation.llm_chain import LLMGenerator
        generator = LLMGenerator()

        tokens = list(generator.generate_stream(
            "麻婆豆腐怎么做", [], "howto", target_dish="麻婆豆腐",
        ))
        assert len(tokens) == 1

    def test_generate_stream_no_api_key(self):
        """Without API key, should fallback to template."""
        saved_key = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            from src_langchain.generation.llm_chain import LLMGenerator
            from langchain_core.documents import Document
            generator = LLMGenerator(api_key=None)

            tokens = list(generator.generate_stream(
                "麻婆豆腐怎么做",
                [Document(page_content="...", metadata={"dish_name": "麻婆豆腐"})],
                "howto",
            ))
            assert len(tokens) >= 1
        finally:
            if saved_key:
                os.environ["DEEPSEEK_API_KEY"] = saved_key


class TestPipelineStream:
    """Integration tests for RAGPipeline.run_stream()."""

    def test_run_stream_returns_stages(self):
        """run_stream should yield rewrite, retrieve, generate, done stages."""
        if not os.environ.get("DEEPSEEK_API_KEY"):
            pytest.skip("DEEPSEEK_API_KEY not set")

        from src_langchain.pipeline import RAGPipeline
        pipe = RAGPipeline(use_llm=True)

        stages_seen = set()
        for event in pipe.run_stream("麻婆豆腐怎么做", top_k=2):
            stages_seen.add(event["stage"])

        assert "rewrite" in stages_seen
        assert "retrieve" in stages_seen
        assert "generate" in stages_seen
        assert "done" in stages_seen

    def test_run_stream_done_has_answer(self):
        """Done stage should include full answer and elapsed time."""
        if not os.environ.get("DEEPSEEK_API_KEY"):
            pytest.skip("DEEPSEEK_API_KEY not set")

        from src_langchain.pipeline import RAGPipeline
        pipe = RAGPipeline(use_llm=True)

        done_event = None
        for event in pipe.run_stream("今天吃什么", top_k=2):
            if event["stage"] == "done":
                done_event = event
                break

        assert done_event is not None
        assert "answer" in done_event
        assert "total_elapsed" in done_event
        assert len(done_event["answer"]) > 0
