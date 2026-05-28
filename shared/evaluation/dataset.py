"""Evaluation dataset loading and validation — shared by both implementations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

log = logging.getLogger("evaluation.dataset")


@dataclass
class EvalSample:
    query: str
    intent: str
    ground_truth: Optional[str] = None
    target_dish: Optional[str] = None
    filters: dict = field(default_factory=dict)

    def __post_init__(self):
        valid_intents = {"recommendation", "howto", "ingredient", "factual"}
        if self.intent not in valid_intents:
            raise ValueError(f"Invalid intent '{self.intent}'")
        if not self.query or not self.query.strip():
            raise ValueError("query must be non-empty")


def load_eval_dataset(path: str) -> List[EvalSample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a YAML list, got {type(raw).__name__}")
    samples = []
    for i, item in enumerate(raw):
        try:
            samples.append(EvalSample(
                query=item["query"],
                intent=item.get("intent", "factual"),
                ground_truth=item.get("ground_truth"),
                target_dish=item.get("target_dish"),
                filters=item.get("filters", {}),
            ))
        except (KeyError, ValueError) as e:
            log.warning("Skipping invalid sample at index %d: %s", i, e)
    log.info("Loaded %d evaluation samples from %s", len(samples), path)
    return samples


def filter_by_intent(samples: List[EvalSample], intent: str) -> List[EvalSample]:
    return [s for s in samples if s.intent == intent]


def samples_by_intent(samples: List[EvalSample]) -> dict:
    groups = {}
    for s in samples:
        groups.setdefault(s.intent, []).append(s)
    return groups


def has_ground_truth(samples: List[EvalSample]) -> List[EvalSample]:
    return [s for s in samples if s.ground_truth]
