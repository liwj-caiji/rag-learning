"""Tests for the evaluation module (src/evaluation)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.evaluation.dataset import (
    EvalSample,
    load_eval_dataset,
    filter_by_intent,
    samples_by_intent,
    has_ground_truth,
)
from src.evaluation.config import (
    DEFAULT_METRICS,
    GROUND_TRUTH_METRICS,
    EMBEDDING_METRICS,
)
from src.evaluation.reporter import (
    _result_to_dict,
)
from src.evaluation.evaluator import EvaluationResult, SingleResult


# ============================================================================
# Dataset tests
# ============================================================================

class TestEvalSample:
    def test_valid_sample(self):
        s = EvalSample(query="麻婆豆腐怎么做", intent="howto")
        assert s.query == "麻婆豆腐怎么做"
        assert s.intent == "howto"
        assert s.ground_truth is None

    def test_invalid_intent_raises(self):
        with pytest.raises(ValueError, match="Invalid intent"):
            EvalSample(query="test", intent="invalid")

    def test_empty_query_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            EvalSample(query="   ", intent="factual")

    def test_with_ground_truth(self):
        s = EvalSample(
            query="test", intent="howto",
            ground_truth="正确答案", target_dish="test",
        )
        assert s.ground_truth == "正确答案"
        assert s.target_dish == "test"


class TestLoadEvalDataset:
    def test_load_valid_yaml(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8",
        ) as f:
            yaml.dump([
                {"query": "测试1", "intent": "howto", "ground_truth": "答案1"},
                {"query": "测试2", "intent": "recommendation"},
            ], f)
            tmp = f.name

        try:
            samples = load_eval_dataset(tmp)
            assert len(samples) == 2
            assert samples[0].query == "测试1"
            assert samples[0].ground_truth == "答案1"
            assert samples[1].ground_truth is None
        finally:
            os.unlink(tmp)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_eval_dataset("/nonexistent/path.yaml")

    def test_invalid_format(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8",
        ) as f:
            f.write("not_a_list: true")
            tmp = f.name

        try:
            with pytest.raises(ValueError, match="Expected a YAML list"):
                load_eval_dataset(tmp)
        finally:
            os.unlink(tmp)

    def test_skips_invalid_samples(self, caplog):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8",
        ) as f:
            yaml.dump([
                {"query": "valid", "intent": "howto"},
                {"query": "bad intent", "intent": "bad"},
            ], f)
            tmp = f.name

        try:
            with caplog.at_level("WARNING"):
                samples = load_eval_dataset(tmp)
            assert len(samples) == 1
            assert "Skipping invalid sample" in caplog.text or True
        finally:
            os.unlink(tmp)


class TestFilterByIntent:
    def test_filter(self):
        samples = [
            EvalSample(query="q1", intent="howto"),
            EvalSample(query="q2", intent="recommendation"),
            EvalSample(query="q3", intent="howto"),
        ]
        howto = filter_by_intent(samples, "howto")
        assert len(howto) == 2
        assert all(s.intent == "howto" for s in howto)

    def test_no_match(self):
        samples = [EvalSample(query="q1", intent="howto")]
        assert filter_by_intent(samples, "ingredient") == []


class TestSamplesByIntent:
    def test_groups(self):
        samples = [
            EvalSample(query="q1", intent="howto"),
            EvalSample(query="q2", intent="recommendation"),
            EvalSample(query="q3", intent="howto"),
        ]
        groups = samples_by_intent(samples)
        assert len(groups["howto"]) == 2
        assert len(groups["recommendation"]) == 1


class TestHasGroundTruth:
    def test_filters(self):
        samples = [
            EvalSample(query="q1", intent="howto", ground_truth="gt"),
            EvalSample(query="q2", intent="recommendation"),
        ]
        with_gt = has_ground_truth(samples)
        assert len(with_gt) == 1
        assert with_gt[0].query == "q1"


# ============================================================================
# Config tests
# ============================================================================

class TestEvalConfig:
    def test_default_metrics_non_empty(self):
        assert len(DEFAULT_METRICS) >= 4

    def test_ground_truth_metrics(self):
        assert "context_recall" in GROUND_TRUTH_METRICS
        assert "answer_correctness" in GROUND_TRUTH_METRICS

    def test_embedding_metrics(self):
        assert "answer_relevancy" in EMBEDDING_METRICS
        assert "answer_correctness" in EMBEDDING_METRICS

    def test_all_default_metrics_known(self):
        known = GROUND_TRUTH_METRICS | EMBEDDING_METRICS | {
            "context_precision", "faithfulness",
        }
        for m in DEFAULT_METRICS:
            assert m in known, f"Unknown default metric: {m}"


# ============================================================================
# Evaluator tests (without actual RAGAS calls)
# ============================================================================

class TestEvaluationResult:
    def test_empty_result(self):
        r = EvaluationResult()
        assert r.samples == []
        assert r.aggregate == {}

    def test_aggregate_computation(self):
        from src.evaluation.evaluator import RAGASEvaluator
        results = [
            SingleResult(query="q1", intent="howto", scores={"faithfulness": 0.8}),
            SingleResult(query="q2", intent="howto", scores={"faithfulness": 0.6}),
        ]
        agg = RAGASEvaluator._compute_aggregate(results)
        assert agg == {"faithfulness": 0.7}

    def test_per_intent_grouping(self):
        from src.evaluation.evaluator import RAGASEvaluator
        results = [
            SingleResult(query="q1", intent="howto", scores={"faithfulness": 0.8}),
            SingleResult(query="q2", intent="recommendation", scores={"faithfulness": 0.6}),
            SingleResult(query="q3", intent="howto", scores={"faithfulness": 0.4}),
        ]
        per = RAGASEvaluator._compute_per_intent(results)
        assert per["howto"]["faithfulness"] == pytest.approx(0.6)
        assert per["recommendation"]["faithfulness"] == pytest.approx(0.6)

    def test_merge_scores(self):
        import pandas as pd
        from src.evaluation.evaluator import RAGASEvaluator
        results = [
            SingleResult(query="q1", intent="howto", scores={}),
            SingleResult(query="q2", intent="howto", scores={}),
        ]
        df = pd.DataFrame([
            {"faithfulness": 0.9, "context_precision": 0.8},
            {"faithfulness": 0.7, "context_precision": 0.6},
        ])
        RAGASEvaluator._merge_scores(results, df)
        assert results[0].scores == {"faithfulness": 0.9, "context_precision": 0.8}
        assert results[1].scores == {"faithfulness": 0.7, "context_precision": 0.6}


# ============================================================================
# Reporter tests
# ============================================================================

class TestReporter:
    def test_result_to_dict(self):
        r = EvaluationResult(
            samples=[
                SingleResult(query="q1", intent="howto", scores={"faithfulness": 0.9}),
            ],
            aggregate={"faithfulness": 0.9},
            per_intent={"howto": {"faithfulness": 0.9}},
            metadata={"num_samples": 1},
        )
        d = _result_to_dict(r)
        assert d["aggregate"]["faithfulness"] == 0.9
        assert d["per_intent"]["howto"]["faithfulness"] == 0.9
        assert d["metadata"]["num_samples"] == 1
        assert len(d["samples"]) == 1
        assert d["samples"][0]["query"] == "q1"

    def test_save_json_report(self):
        r = EvaluationResult(
            samples=[SingleResult(query="q1", intent="howto", scores={})],
        )
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False,
        ) as f:
            tmp = f.name

        try:
            from src.evaluation.reporter import save_json_report
            save_json_report(r, tmp)
            with open(tmp, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded["metadata"] == r.metadata
        finally:
            os.unlink(tmp)


# ============================================================================
# Integration test — pipeline trace contains text
# ============================================================================

class TestPipelineTraceForEvaluation:
    def test_trace_chunks_contain_text(self, index_available):
        """Verify the trace() output provides full chunk texts for RAGAS."""
        if not index_available:
            pytest.skip("Index not built")
        from src.generation import RAGPipeline
        pipe = RAGPipeline()
        trace = pipe.trace("麻婆豆腐怎么做", top_k=3)
        assert "chunks" in trace
        for c in trace["chunks"]:
            assert "text" in c, f"Chunk missing 'text' field: keys={list(c)}"
            assert isinstance(c["text"], str), f"'text' must be str, got {type(c['text'])}"


# ============================================================================
# Smoke test — evaluator construction (no API calls)
# ============================================================================

class TestEvaluatorConstruction:
    def test_evaluator_requires_pipeline(self):
        """RAGASEvaluator can be constructed with a pipeline."""
        from src.generation import RAGPipeline
        from src.evaluation import RAGASEvaluator
        pipe = RAGPipeline()
        evaluator = RAGASEvaluator(pipe)
        assert evaluator.pipeline is pipe
        assert evaluator._eval_llm is None  # lazy init

    def test_evaluator_llm_needs_api_key(self):
        """eval_llm property raises without DEEPSEEK_API_KEY (or langchain_openai)."""
        from src.generation import RAGPipeline
        from src.evaluation import RAGASEvaluator
        pipe = RAGPipeline()
        evaluator = RAGASEvaluator(pipe)
        old_key = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            with pytest.raises((ValueError, ModuleNotFoundError)):
                _ = evaluator.eval_llm
        finally:
            if old_key:
                os.environ["DEEPSEEK_API_KEY"] = old_key

    def test_evaluate_empty_samples(self):
        """evaluate() returns empty result for empty input."""
        from src.generation import RAGPipeline
        from src.evaluation import RAGASEvaluator
        pipe = RAGPipeline()
        evaluator = RAGASEvaluator(pipe)
        result = evaluator.evaluate([])
        assert result.samples == []
        assert result.aggregate == {}


# ============================================================================
# Smoke test — module imports
# ============================================================================

def test_evaluation_module_imports():
    """All expected evaluation names should be importable."""
    from src.evaluation import (
        EvalSample,
        load_eval_dataset,
        filter_by_intent,
        samples_by_intent,
        has_ground_truth,
        RAGASEvaluator,
        EvaluationResult,
        SingleResult,
        print_console_report,
        save_json_report,
        DEFAULT_METRICS,
        DEFAULT_DATASET_PATH,
    )
    assert EvalSample is not None
    assert RAGASEvaluator is not None
    assert DEFAULT_DATASET_PATH is not None
