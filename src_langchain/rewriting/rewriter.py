"""Query rewriter factory."""

from __future__ import annotations

from typing import Optional

from .rule_classifier import RuleIntentClassifier
from .llm_classifier import LLMIntentClassifier


def get_intent_classifier(
    use_llm: bool = False,
    model: str = None,
    api_base: str = None,
    callbacks: Optional[list] = None,
):
    if use_llm:
        kwargs = {}
        if model:
            kwargs["model"] = model
        if api_base:
            kwargs["api_base"] = api_base
        if callbacks:
            kwargs["callbacks"] = callbacks
        return LLMIntentClassifier(**kwargs)
    return RuleIntentClassifier()
