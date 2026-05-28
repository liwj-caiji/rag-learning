"""Evaluation configuration — shared by both implementations."""

import os

LLM_API_KEY_ENV = "DEEPSEEK_API_KEY"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = os.path.join(BASE_DIR, "data", "evaluation")
DEFAULT_DATASET_PATH = os.path.join(EVAL_DIR, "test_queries.yaml")

EVAL_LLM_MODEL = "deepseek-chat"
EVAL_LLM_API_BASE = "https://api.deepseek.com"
EVAL_LLM_TEMPERATURE = 0.0
EVAL_LLM_MAX_TOKENS = 8192

DEFAULT_METRICS = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
]

GROUND_TRUTH_METRICS = {"context_precision", "context_recall", "answer_correctness"}
EMBEDDING_METRICS = {"answer_relevancy", "answer_correctness"}
EVAL_BATCH_SIZE = 5
