#!/usr/bin/env python3
"""CLI entry point for RAGAS evaluation.

Usage:
    python scripts/evaluate.py --mode llm
    python scripts/evaluate.py --mode rule --limit 5
    python scripts/evaluate.py --backend langchain --limit 5
    python scripts/evaluate.py --intent howto --output report.json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.evaluation import (
    load_eval_dataset,
    filter_by_intent,
    RAGASEvaluator,
    print_console_report,
    save_json_report,
    DEFAULT_METRICS,
    DEFAULT_DATASET_PATH,
    LLM_API_KEY_ENV,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the recipe RAG pipeline with RAGAS metrics.",
    )
    parser.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET_PATH,
        help="Path to evaluation dataset YAML file.",
    )
    parser.add_argument(
        "--backend", type=str, choices=("src", "langchain"), default="src",
        help="Backend implementation (default: src).",
    )
    parser.add_argument(
        "--mode", type=str, choices=("rule", "llm"), default="rule",
        help="Pipeline mode: rule (template) or llm (DeepSeek).",
    )
    parser.add_argument(
        "--intent", type=str, default=None,
        choices=("recommendation", "howto", "ingredient", "factual"),
        help="Only evaluate samples of a specific intent.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit evaluation to the first N samples (0 = all).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save JSON report to this path.",
    )
    parser.add_argument(
        "--metrics", type=str, nargs="*", default=DEFAULT_METRICS,
        help="RAGAS metrics to compute (default: all).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=5,
        help="Batch size for RAGAS evaluate().",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Enable DEBUG logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("evaluate")

    # Load dataset
    log.info("Loading dataset: %s", args.dataset)
    samples = load_eval_dataset(args.dataset)
    if args.intent:
        samples = filter_by_intent(samples, args.intent)
        log.info("Filtered to intent=%s: %d samples", args.intent, len(samples))
    if args.limit > 0:
        samples = samples[:args.limit]
        log.info("Limited to first %d samples", args.limit)

    if not samples:
        print("No samples to evaluate.")
        sys.exit(1)

    # Build pipeline
    use_llm = args.mode == "llm"
    if args.backend == "langchain":
        from src_langchain.pipeline import RAGPipeline
    else:
        from src.generation import RAGPipeline
    log.info("Creating pipeline: backend=%s mode=%s", args.backend, args.mode)
    try:
        pipeline = RAGPipeline(use_llm=use_llm)
    except ValueError as e:
        log.error("Failed to create LLM pipeline: %s", e)
        log.error("Make sure %s is set. Falling back to rule mode.", LLM_API_KEY_ENV)
        pipeline = RAGPipeline(use_llm=False)

    # Evaluate
    evaluator = RAGASEvaluator(pipeline)
    log.info("Running evaluation on %d samples with metrics: %s",
             len(samples), args.metrics)
    result = evaluator.evaluate(samples, metrics=args.metrics, batch_size=args.batch_size)

    # Report
    print_console_report(result)

    if args.output:
        save_json_report(result, args.output)
        log.info("Report saved to %s", args.output)


if __name__ == "__main__":
    main()
