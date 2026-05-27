"""RAGAS-based evaluation for the recipe RAG pipeline."""

from .dataset import EvalSample, load_eval_dataset, filter_by_intent, samples_by_intent, has_ground_truth
from .evaluator import RAGASEvaluator, EvaluationResult, SingleResult
from .reporter import print_console_report, save_json_report
from .config import DEFAULT_METRICS, DEFAULT_DATASET_PATH

__all__ = [
    "EvalSample",
    "load_eval_dataset",
    "filter_by_intent",
    "samples_by_intent",
    "has_ground_truth",
    "RAGASEvaluator",
    "EvaluationResult",
    "SingleResult",
    "print_console_report",
    "save_json_report",
    "DEFAULT_METRICS",
    "DEFAULT_DATASET_PATH",
]
