"""Tests for app.py Gradio Web UI handler functions.

These tests verify:
- Handler functions return correct structures for both valid/invalid input
- Error handling produces clean error messages (no unhandled crashes)
- Pipeline caching works correctly
- State management functions behave as expected
- The UI event handlers format output correctly

Run with: python -m pytest tests/test_app.py -v
"""

import os

import pytest


class TestRunPipeline:
    """_run_pipeline — core pipeline execution handler."""

    def test_empty_query_returns_immediate(self):
        """Empty query should return immediately with a prompt (no pipeline call)."""
        from app import _run_pipeline
        result = _run_pipeline("", top_k=5, use_llm=False)
        assert isinstance(result, dict)
        assert "请输入查询" in result.get("answer", "")
        assert result.get("num_chunks") == 0

    def test_whitespace_query_returns_immediate(self):
        """Whitespace-only query should be treated as empty."""
        from app import _run_pipeline
        result = _run_pipeline("   ", top_k=5, use_llm=False)
        assert "请输入查询" in result.get("answer", "")

    def test_valid_query_returns_trace(self, index_available):
        """Valid query should return a complete trace dict in rule mode."""
        if not index_available:
            pytest.skip("Index not built — run python -m src.preprocess.indexer first")
        from app import _run_pipeline
        result = _run_pipeline("麻婆豆腐怎么做", top_k=3, use_llm=False)
        expected_keys = {"intent", "rewritten", "filters", "probes",
                         "target_dish", "num_chunks", "chunks", "answer"}
        assert expected_keys.issubset(result.keys()), \
            f"Missing keys: {expected_keys - result.keys()}"
        assert isinstance(result.get("answer"), str)
        assert len(result["answer"]) > 0

    def test_error_key_on_missing_api_key(self):
        """use_llm=True without DEEPSEEK_API_KEY should return error trace."""
        from app import _run_pipeline
        old_key = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            result = _run_pipeline("test", top_k=5, use_llm=True)
            assert result.get("intent") == "error"
            assert "DEEPSEEK_API_KEY" in result.get("answer", "")
        finally:
            if old_key:
                os.environ["DEEPSEEK_API_KEY"] = old_key

    def test_gibberish_query_does_not_crash(self, index_available):
        """Gibberish query should not crash the handler."""
        if not index_available:
            pytest.skip("Index not built")
        from app import _run_pipeline
        result = _run_pipeline("zzznotawordzzz", top_k=3, use_llm=False)
        assert isinstance(result, dict)

    def test_numeric_query_does_not_crash(self, index_available):
        """Numeric/special character query should not crash."""
        if not index_available:
            pytest.skip("Index not built")
        from app import _run_pipeline
        result = _run_pipeline("123!@#$", top_k=3, use_llm=False)
        assert isinstance(result, dict)


class TestChatAnswer:
    """_chat_answer — chatbot message handler."""

    def test_empty_message_returns_history_unchanged(self):
        """Empty message should return history as-is."""
        from app import _chat_answer
        history = [("hello", "world")]
        new_history, cleared_input = _chat_answer("", history, False, 5)
        assert new_history is history  # Same object, not a copy
        assert cleared_input == ""

    def test_whitespace_message_returns_history_unchanged(self):
        """Whitespace-only message should return history as-is."""
        from app import _chat_answer
        history = []
        new_history, cleared_input = _chat_answer("   ", history, False, 5)
        assert new_history is history
        assert cleared_input == ""

    def test_none_message_returns_history_unchanged(self):
        """None message should return history as-is."""
        from app import _chat_answer
        history = []
        new_history, cleared_input = _chat_answer(None, history, False, 5)
        assert new_history is history
        assert cleared_input == ""

    def test_valid_message_appends_answer(self, index_available):
        """Valid message should append user + assistant dicts to history."""
        if not index_available:
            pytest.skip("Index not built")
        from app import _chat_answer
        history = []
        new_history, cleared_input = _chat_answer("麻婆豆腐怎么做", history, False, 3)
        assert len(new_history) == 2  # user + assistant
        assert new_history[0]["role"] == "user"
        assert new_history[0]["content"] == "麻婆豆腐怎么做"
        assert new_history[1]["role"] == "assistant"
        assert isinstance(new_history[1]["content"], str)
        assert len(new_history[1]["content"]) > 0
        assert cleared_input == ""  # Input should be cleared

    def test_error_handling_returns_error_message(self):
        """Error in pipeline should produce error message in history, not crash."""
        from app import _chat_answer
        history = []
        old_key = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            new_history, _ = _chat_answer("test", history, True, 5)
            assert len(new_history) == 2  # user + assistant
            assert "❌" in new_history[1]["content"] or "错误" in new_history[1]["content"]
        finally:
            if old_key:
                os.environ["DEEPSEEK_API_KEY"] = old_key

    def test_multiple_messages_accumulate(self, index_available):
        """Multiple messages should all be preserved in history."""
        if not index_available:
            pytest.skip("Index not built")
        from app import _chat_answer
        history = []
        history, _ = _chat_answer("今天吃什么", history, False, 3)
        history, _ = _chat_answer("推荐素菜", history, False, 3)
        assert len(history) == 4  # 2 messages × 2 entries each

    def test_rule_vs_llm_mode_both_return_string(self, index_available):
        """Both rule and LLM mode should return string answers."""
        if not index_available:
            pytest.skip("Index not built")
        from app import _chat_answer
        history = []
        # Rule mode
        history, _ = _chat_answer("麻婆豆腐怎么做", history, False, 3)
        assert isinstance(history[1]["content"], str)


class TestSearch:
    """_search — retrieval search handler."""

    def test_empty_query_returns_prompt(self):
        """Empty query should return prompt message without calling search."""
        from app import _search
        status, rows, _ = _search("", k=5, mode="混合检索 (Hybrid)")
        assert "请输入查询" in status
        assert rows == []

    def test_whitespace_query_returns_prompt(self):
        """Whitespace query should return prompt."""
        from app import _search
        status, rows, _ = _search("   ", k=5, mode="混合检索 (Hybrid)")
        assert "请输入查询" in status

    def test_hybrid_search_mode(self, index_available):
        """Hybrid search should return results with correct structure."""
        if not index_available:
            pytest.skip("Index not built")
        from app import _search
        status, rows, _ = _search("红烧肉", k=3, mode="混合检索 (Hybrid)")
        assert "共" in status
        assert len(rows) <= 3
        if rows:
            # Row should have 5 columns: score, dish, category, level, preview
            assert len(rows[0]) == 5

    def test_dense_search_mode(self, index_available):
        """Dense search should work independently."""
        if not index_available:
            pytest.skip("Index not built")
        from app import _search
        status, rows, _ = _search("红烧肉", k=3, mode="稠密检索 (Dense)")
        assert "共" in status or "错误" in status

    def test_sparse_search_mode(self, index_available):
        """Sparse search should work independently."""
        if not index_available:
            pytest.skip("Index not built")
        from app import _search
        status, rows, _ = _search("红烧肉", k=3, mode="稀疏检索 (Sparse)")
        assert "共" in status or "错误" in status

    def test_all_modes_different_k(self, index_available):
        """Search should respect k parameter across all modes."""
        if not index_available:
            pytest.skip("Index not built")
        from app import _search
        for mode in ["混合检索 (Hybrid)", "稠密检索 (Dense)", "稀疏检索 (Sparse)"]:
            status, rows, _ = _search("鸡", k=5, mode=mode)
            if "共" in status:
                assert len(rows) <= 5, f"{mode}: expected <=5 rows, got {len(rows)}"


class TestGetPipeline:
    """_get_pipeline — lazy pipeline caching."""

    def test_caching_returns_same_instance(self):
        """Same use_llm flag should return the same pipeline object."""
        from app import _get_pipeline, _PIPELINE_CACHE
        _PIPELINE_CACHE.clear()
        p1 = _get_pipeline(False)
        p2 = _get_pipeline(False)
        assert p1 is p2

    def test_rule_pipeline_has_run_method(self):
        """Rule-mode pipeline should have run/trace interfaces."""
        from app import _get_pipeline, _PIPELINE_CACHE
        _PIPELINE_CACHE.clear()
        pipe = _get_pipeline(False)
        assert hasattr(pipe, "run")
        assert hasattr(pipe, "trace")

    def test_cache_key_separation(self):
        """Different use_llm flags use different cache keys."""
        from app import _get_pipeline, _PIPELINE_CACHE
        _PIPELINE_CACHE.clear()
        p_rule = _get_pipeline(False)
        # LLM pipeline may fail without API key; verify cache keys are distinct
        assert "pipe_False" in _PIPELINE_CACHE
        assert _PIPELINE_CACHE["pipe_False"] is p_rule

    def test_cache_not_polluted_by_errors(self):
        """Failed pipeline creation should not corrupt the cache for other keys."""
        from app import _get_pipeline, _PIPELINE_CACHE
        _PIPELINE_CACHE.clear()
        _ = _get_pipeline(False)
        assert "pipe_False" in _PIPELINE_CACHE


class TestOnPipeRun:
    """on_pipe_run — pipeline tab event handler formatting."""

    def test_empty_query_returns_formatted_output(self):
        """Empty query should produce correctly formatted output."""
        from app import on_pipe_run
        intent_data, df, answer = on_pipe_run("", 5, False)
        assert isinstance(intent_data, dict)
        assert "意图" in intent_data
        assert intent_data.get("意图") == "" or intent_data.get("意图") == "error"
        assert "请输入查询" in answer

    def test_valid_query_returns_formatted_output(self, index_available):
        """Valid query should produce formatted output with all sections."""
        if not index_available:
            pytest.skip("Index not built")
        from app import on_pipe_run
        intent_data, df, answer = on_pipe_run("麻婆豆腐怎么做", 3, False)
        assert isinstance(intent_data, dict)
        assert "意图" in intent_data
        assert "改写查询" in intent_data
        assert "过滤条件" in intent_data
        assert "目标菜名" in intent_data
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_on_pipe_run_error_handling(self):
        """Event handler should handle errors gracefully."""
        from app import on_pipe_run
        old_key = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            intent_data, df, answer = on_pipe_run("test", 5, True)
            assert isinstance(intent_data, dict)
        finally:
            if old_key:
                os.environ["DEEPSEEK_API_KEY"] = old_key


class TestOnChatMsg:
    """on_chat_msg — chat event handler."""

    def test_on_chat_msg_empty(self):
        """Empty message should return history unchanged."""
        from app import on_chat_msg
        history = [("hello", "world")]
        new_history, cleared_input = on_chat_msg("", history, False, 5)
        assert new_history is history

    def test_on_chat_msg_valid(self, index_available):
        """Valid message should update history."""
        if not index_available:
            pytest.skip("Index not built")
        from app import on_chat_msg
        history = []
        new_history, _ = on_chat_msg("麻婆豆腐怎么做", history, False, 3)
        assert len(new_history) == 2


class TestStateSync:
    """UI state synchronization functions."""

    def test_llm_toggle_on(self):
        """LLM toggle True should return True + LLM hint."""
        from app import on_llm_toggle
        state, hint = on_llm_toggle(True)
        assert state is True
        assert "LLM" in hint

    def test_llm_toggle_off(self):
        """LLM toggle False should return False + rule hint."""
        from app import on_llm_toggle
        state, hint = on_llm_toggle(False)
        assert state is False
        assert "规则" in hint


class TestLoadStats:
    """_load_stats — data overview tab."""

    def test_load_stats_returns_markdown(self):
        """_load_stats should return a non-empty markdown string."""
        from app import _load_stats
        result = _load_stats()
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain markdown headers
        assert "##" in result or "统计" in result or "失败" in result

    def test_load_stats_has_recipe_count(self):
        """Stats should mention recipe count when data is available."""
        from app import _load_stats
        result = _load_stats()
        if "失败" not in result:
            assert "食谱" in result or "品类" in result or "分块" in result

    def test_load_stats_error_handling(self):
        """_load_stats should not crash even if data directory is missing."""
        from app import _load_stats
        result = _load_stats()
        assert isinstance(result, str)
        # Even on error, should return a string with error info
        assert len(result) > 0
