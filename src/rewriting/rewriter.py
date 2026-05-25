"""查询改写器：规则驱动 + 预留 LLM 扩展接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from src.config import LLM_INTENT_MODEL
from .intent import IntentResult, classify_intent


class BaseQueryRewriter(ABC):
    """Abstract query rewriter."""

    @abstractmethod
    def rewrite(self, query: str) -> IntentResult:
        ...


class RuleQueryRewriter(BaseQueryRewriter):
    """Rule-based query rewriter.

    Zero external dependencies, works offline.
    """

    def rewrite(self, query: str) -> IntentResult:
        return classify_intent(query)


class LLMQueryRewriter(BaseQueryRewriter):
    """LLM-powered query rewriter (future extension)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = LLM_INTENT_MODEL,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model

    def rewrite(self, query: str) -> IntentResult:
        # TODO: call LLM with structured output prompt
        return classify_intent(query)


# Default alias for convenience
QueryRewriter = RuleQueryRewriter
