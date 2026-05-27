"""Core evaluator wrapping RAGAS for the recipe RAG pipeline."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import LLM_API_KEY_ENV
from .config import (
    DEFAULT_METRICS,
    EMBEDDING_METRICS,
    EVAL_LLM_MODEL,
    EVAL_LLM_API_BASE,
    EVAL_LLM_TEMPERATURE,
    EVAL_LLM_MAX_TOKENS,
    EVAL_BATCH_SIZE,
)
from .dataset import EvalSample

log = logging.getLogger("evaluation.evaluator")

_DEFAULT_MODEL_NAME = "shibing624/text2vec-base-chinese"


class _EmbeddingsAdapter:
    """Wraps ragas native embeddings to also expose LangChain's embed_query."""

    def __init__(self, ragas_embeddings):
        self._wrapped = ragas_embeddings

    def embed_text(self, text: str) -> List[float]:
        return self._wrapped.embed_text(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self._wrapped.embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._wrapped.embed_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._wrapped.embed_texts(texts)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


@dataclass
class SingleResult:
    """Evaluation result for a single sample."""

    query: str
    intent: str
    scores: Dict[str, float]
    answer: str = ""
    num_chunks: int = 0
    contexts: List[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Aggregated evaluation result."""

    samples: List[SingleResult] = field(default_factory=list)
    aggregate: Dict[str, float] = field(default_factory=dict)
    per_intent: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGASEvaluator:
    """Evaluates the RAG pipeline using RAGAS metrics.

    Usage:
        from src.generation import RAGPipeline
        pipeline = RAGPipeline(use_llm=True)
        evaluator = RAGASEvaluator(pipeline)
        result = evaluator.evaluate(samples)
        print(result.aggregate)
    """

    def __init__(
        self,
        pipeline,
        llm_model: str = EVAL_LLM_MODEL,
        llm_api_base: str = EVAL_LLM_API_BASE,
        embed_model: str = _DEFAULT_MODEL_NAME,
    ):
        self.pipeline = pipeline
        self._llm_model = llm_model
        self._llm_api_base = llm_api_base
        self._embed_model = embed_model
        self._eval_llm = None
        self._eval_embeddings = None

    @property
    def eval_llm(self):
        """Lazy-init a ragas-compatible InstructorLLM via llm_factory."""
        if self._eval_llm is None:
            import openai
            from ragas.llms import llm_factory
            api_key = os.environ.get(LLM_API_KEY_ENV)
            if not api_key:
                raise ValueError(
                    f"{LLM_API_KEY_ENV} environment variable is required for RAGAS evaluation"
                )
            client = openai.OpenAI(api_key=api_key, base_url=self._llm_api_base)
            self._eval_llm = llm_factory(
                self._llm_model,
                client=client,
                max_tokens=EVAL_LLM_MAX_TOKENS,
            )
        return self._eval_llm

    @property
    def eval_embeddings(self):
        """Lazy-init embeddings with both ragas-native and LangChain interfaces."""
        if self._eval_embeddings is None:
            from ragas.embeddings import HuggingFaceEmbeddings as RagasHFE
            native = RagasHFE(model=self._embed_model)
            self._eval_embeddings = _EmbeddingsAdapter(native)
        return self._eval_embeddings

    def evaluate(
        self,
        samples: List[EvalSample],
        metrics: Optional[List[str]] = None,
        batch_size: int = EVAL_BATCH_SIZE,
    ) -> EvaluationResult:
        """Run RAGAS evaluation on a list of samples.

        Args:
            samples: List of evaluation samples.
            metrics: List of RAGAS metric names. Defaults to all.
            batch_size: Number of samples per RAGAS evaluate() call.

        Returns:
            EvaluationResult with per-sample scores and aggregates.
        """
        if not samples:
            log.warning("No samples provided for evaluation")
            return EvaluationResult()

        metrics = metrics or DEFAULT_METRICS
        ragas_metrics = self._build_metrics(metrics)

        # Collect pipeline outputs for each sample
        rows = []
        single_results = []
        for sample in samples:
            sr = self._evaluate_one(sample)
            single_results.append(sr)
            row = {
                "user_input": sample.query,
                "response": sr.answer,
                "retrieved_contexts": sr.contexts,
                "reference": sample.ground_truth or "",
            }
            rows.append(row)

        # Build RAGAS Dataset and run evaluation
        ds = self._to_dataset(rows)
        if ds is None:
            log.error("Failed to build RAGAS dataset")
            return EvaluationResult(samples=single_results)

        try:
            from ragas import evaluate as ragas_evaluate
            ragas_result = ragas_evaluate(
                dataset=ds,
                metrics=ragas_metrics,
                llm=self.eval_llm,
                embeddings=self.eval_embeddings,
            )
            scores_df = ragas_result.to_pandas()
        except Exception as e:
            log.error("RAGAS evaluate() failed: %s", e)
            return EvaluationResult(samples=single_results)

        # Merge RAGAS scores back into SingleResults
        self._merge_scores(single_results, scores_df)

        # Compute aggregates
        aggregate = self._compute_aggregate(single_results)
        per_intent = self._compute_per_intent(single_results)

        return EvaluationResult(
            samples=single_results,
            aggregate=aggregate,
            per_intent=per_intent,
            metadata={
                "num_samples": len(samples),
                "metrics_used": metrics,
                "llm_model": self._llm_model,
            },
        )

    def _evaluate_one(self, sample: EvalSample) -> SingleResult:
        """Run pipeline.trace() for one sample and extract data for RAGAS."""
        try:
            trace = self.pipeline.trace(sample.query)
            context_texts = [
                c.get("text", "") for c in trace.get("chunks", [])
            ]
            return SingleResult(
                query=sample.query,
                intent=trace.get("intent", sample.intent),
                scores={},
                answer=trace.get("answer", ""),
                num_chunks=len(context_texts),
                contexts=context_texts,
            )
        except Exception as e:
            log.warning("Pipeline trace failed for %r: %s", sample.query, e)
            return SingleResult(
                query=sample.query,
                intent=sample.intent,
                scores={"error": 1.0},
                answer="",
                num_chunks=0,
            )

    def _build_metrics(self, metric_names: List[str]):
        """Instantiate RAGAS metric objects with Chinese prompts via adapt_prompts."""
        from ragas.metrics import (
            ContextPrecision,
            ContextRecall,
            Faithfulness,
            AnswerRelevancy,
            AnswerCorrectness,
        )

        metric_map = {
            "context_precision": ContextPrecision,
            "context_recall": ContextRecall,
            "faithfulness": Faithfulness,
            "answer_relevancy": AnswerRelevancy,
            "answer_correctness": AnswerCorrectness,
        }

        built = []
        for name in metric_names:
            if name not in metric_map:
                log.warning("Unknown metric: %s", name)
                continue
            cls = metric_map[name]
            kwargs: Dict[str, Any] = {}
            if name in EMBEDDING_METRICS:
                kwargs["embeddings"] = self.eval_embeddings
            built.append(cls(llm=self.eval_llm, **kwargs))

        if built:
            self._adapt_prompts_chinese(built)
        return built

    def _adapt_prompts_chinese(self, metrics_list):
        """Adapt metric prompts to Chinese via ragas adapt_prompts, with disk cache.

        On first run, calls the evaluator LLM to translate each metric's prompts
        to Chinese and caches them to disk. Subsequent runs load from cache.
        """
        import asyncio
        import pickle
        from pathlib import Path

        cache_dir = Path(__file__).parent / ".ragas_cache"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / "chinese_prompts.pkl"

        # Try loading cached prompts
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                for metric, prompts in zip(metrics_list, cached):
                    metric.set_prompts(**prompts)
                log.info("Loaded %d Chinese prompts from cache", len(cached))
                return
            except Exception as e:
                log.warning("Cache load failed (%s), re-adapting", e)
                cache_file.unlink(missing_ok=True)

        async def _adapt():
            for metric in metrics_list:
                adapted = await metric.adapt_prompts(language="chinese", llm=self.eval_llm)
                metric.set_prompts(**adapted)

        log.info("Adapting %d metrics to Chinese via LLM...", len(metrics_list))
        asyncio.run(_adapt())

        # Persist adapted prompts to disk cache
        try:
            prompts_data = [metric.get_prompts() for metric in metrics_list]
            with open(cache_file, "wb") as f:
                pickle.dump(prompts_data, f)
            log.info("Chinese prompts cached to %s", cache_file)
        except Exception as e:
            log.warning("Failed to cache prompts: %s", e)

    @staticmethod
    def _to_dataset(rows: List[dict]):
        """Convert collected rows to a RAGAS-compatible Dataset."""
        try:
            from datasets import Dataset
            return Dataset.from_pandas(pd.DataFrame(rows))
        except Exception as e:
            log.error("Failed to build Dataset: %s", e)
            return None

    @staticmethod
    def _merge_scores(
        results: List[SingleResult], scores_df: pd.DataFrame
    ):
        """Merge RAGAS scores back into SingleResult objects."""
        # Identify metric columns: exclude known input/internal columns
        exclude = {"question", "answer", "contexts", "ground_truth",
                   "user_input", "retrieved_contexts", "reference", "response"}
        score_cols = [c for c in scores_df.columns if c not in exclude
                      and not c.startswith("_")]
        # Only keep columns with numeric values
        metric_cols = []
        for col in score_cols:
            try:
                val = scores_df[col].iloc[0]
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    float(val)
                    metric_cols.append(col)
            except (ValueError, TypeError):
                pass

        for i, sr in enumerate(results):
            if i < len(scores_df):
                row = scores_df.iloc[i]
                for col in metric_cols:
                    val = row[col]
                    if val is not None and not (isinstance(val, float) and pd.isna(val)):
                        sr.scores[col] = float(val)

    @staticmethod
    def _compute_aggregate(results: List[SingleResult]) -> Dict[str, float]:
        """Compute mean score per metric across all samples."""
        metric_values: Dict[str, List[float]] = {}
        for sr in results:
            for name, score in sr.scores.items():
                metric_values.setdefault(name, []).append(score)

        return {
            name: sum(vals) / len(vals)
            for name, vals in metric_values.items()
            if vals
        }

    @staticmethod
    def _compute_per_intent(
        results: List[SingleResult],
    ) -> Dict[str, Dict[str, float]]:
        """Compute mean scores grouped by intent."""
        grouped: Dict[str, Dict[str, List[float]]] = {}
        for sr in results:
            intent_metrics = grouped.setdefault(sr.intent, {})
            for name, score in sr.scores.items():
                intent_metrics.setdefault(name, []).append(score)

        return {
            intent: {
                name: sum(vals) / len(vals)
                for name, vals in metric_vals.items()
                if vals
            }
            for intent, metric_vals in grouped.items()
        }
