"""Format and output evaluation results."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

from .evaluator import EvaluationResult, SingleResult

log = logging.getLogger("evaluation.reporter")


def print_console_report(result: EvaluationResult):
    """Print evaluation results as a formatted table in the console."""
    if not result.samples:
        print("No evaluation results to display.")
        return

    meta = result.metadata
    print("=" * 64)
    print("  RAGAS Evaluation Report")
    print("=" * 64)
    print(f"  Samples:     {meta.get('num_samples', len(result.samples))}")
    print(f"  LLM model:   {meta.get('llm_model', 'N/A')}")
    print(f"  Metrics:     {', '.join(meta.get('metrics_used', []))}")
    print("-" * 64)

    # Aggregates
    if result.aggregate:
        print("\n  Overall Scores:")
        print(f"  {'Metric':<30} {'Score':>8}")
        print(f"  {'-'*30} {'-'*8}")
        for name, score in sorted(result.aggregate.items()):
            print(f"  {name:<30} {score:>8.4f}")

    # Per-intent breakdown
    if result.per_intent:
        print(f"\n  By Intent:")
        all_metrics = sorted(result.aggregate.keys())
        if not all_metrics:
            intents = list(result.per_intent.keys())
            print(f"  {'Intent':<20}", end="")
            for m in all_metrics or []:
                print(f"  {m[:10]:>10}", end="")
            print()
            for intent, scores in sorted(result.per_intent.items()):
                print(f"  {intent:<20}", end="")
                for m in all_metrics or []:
                    val = scores.get(m, float("nan"))
                    print(f"  {val:>10.4f}", end="")
                print()
        else:
            cols = sorted(all_metrics)
            # Header
            header = f"  {'Intent':<20}"
            for c in cols:
                header += f"  {c[:10]:>10}"
            print(header)
            print(f"  {'-'*20}", end="")
            for _ in cols:
                print(f"  {'-'*10}", end="")
            print()
            # Rows
            for intent in ["recommendation", "howto", "ingredient", "factual"]:
                scores = result.per_intent.get(intent)
                if not scores:
                    continue
                row = f"  {intent:<20}"
                for c in cols:
                    val = scores.get(c, float("nan"))
                    row += f"  {val:>10.4f}"
                print(row)

    # Per-sample detail
    print(f"\n  Per-Sample Scores:")
    print(f"  {'#':<4} {'Query':<30} {'Intent':<16}", end="")
    metric_names = sorted(result.aggregate.keys()) if result.aggregate else []
    for m in metric_names:
        print(f"  {m[:8]:>8}", end="")
    print()
    print(f"  {'-'*4} {'-'*30} {'-'*16}", end="")
    for _ in metric_names:
        print(f"  {'-'*8}", end="")
    print()

    for i, sr in enumerate(result.samples):
        q = sr.query[:28] + ".." if len(sr.query) > 30 else sr.query
        print(f"  {i:>4} {q:<30} {sr.intent:<16}", end="")
        for m in metric_names:
            val = sr.scores.get(m, float("nan"))
            print(f"  {val:>8.4f}", end="")
        print()

    print("=" * 64)


def save_json_report(result: EvaluationResult, path: str):
    """Save evaluation result as a JSON file."""
    data = _result_to_dict(result)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info("JSON report saved to %s", path)


def _result_to_dict(result: EvaluationResult) -> dict:
    """Convert EvaluationResult to a JSON-serializable dict."""
    return {
        "aggregate": result.aggregate,
        "per_intent": result.per_intent,
        "metadata": result.metadata,
        "samples": [
            {
                "query": sr.query,
                "intent": sr.intent,
                "answer": sr.answer[:500] if sr.answer else "",
                "scores": sr.scores,
                "num_chunks": sr.num_chunks,
            }
            for sr in result.samples
        ],
    }

METRIC_LABELS_ZH: Dict[str, str] = {
    "context_precision": "上下文精确度",
    "context_recall": "上下文召回率",
    "faithfulness": "忠实度",
    "answer_relevancy": "回答相关性",
    "answer_correctness": "回答正确性",
}
