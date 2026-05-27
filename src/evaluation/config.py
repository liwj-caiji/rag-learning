"""Evaluation-specific configuration."""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = os.path.join(BASE_DIR, "data", "evaluation")
DEFAULT_DATASET_PATH = os.path.join(EVAL_DIR, "test_queries.yaml")

# LLM used by RAGAS for metric computation (OpenAI-compatible)
# Use non-reasoning model for reliable structured JSON output
EVAL_LLM_MODEL = "deepseek-chat"
EVAL_LLM_API_BASE = "https://api.deepseek.com"
EVAL_LLM_TEMPERATURE = 0.0
EVAL_LLM_MAX_TOKENS = 8192

# RAGAS metrics to compute by default
DEFAULT_METRICS = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
]

# Metrics that require ground_truth (skip when unavailable)
GROUND_TRUTH_METRICS = {"context_precision", "context_recall", "answer_correctness"}

# Metrics that need embeddings
EMBEDDING_METRICS = {"answer_relevancy", "answer_correctness"}

# Batch size for RAGAS evaluate()
EVAL_BATCH_SIZE = 5
