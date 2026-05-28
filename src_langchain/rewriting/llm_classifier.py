"""LLM-powered intent classifier using LangChain ChatOpenAI with structured output."""

import os
from typing import Dict, Optional, List

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..config import (
    LLM_INTENT_MODEL, LLM_INTENT_API_BASE, LLM_INTENT_TIMEOUT,
    LLM_INTENT_TEMPERATURE, LLM_API_KEY_ENV,
)
from .intent_types import IntentResult, IntentType, build_probes
from .rule_classifier import classify_intent


class IntentFilterSchema(BaseModel):
    difficulty: Optional[str] = Field(None, description="难度: ★ / ★★ / ★★★★")
    category: Optional[str] = Field(None, description="菜品分类")
    calories: Optional[str] = Field(None, description="热量: low / high")


class IntentClassificationSchema(BaseModel):
    intent: str = Field(..., description="recommendation | howto | ingredient | factual")
    rewritten_query: str = Field("", description="改写后的搜索查询")
    filters: IntentFilterSchema = Field(default_factory=IntentFilterSchema)
    target_dish: Optional[str] = Field(None, description="目标菜名")
    probes: Optional[List[str]] = Field(None, description="搜索探针")
    confidence: float = Field(0.8, description="置信度 0.0-1.0")


INTENT_SYSTEM_PROMPT = """你是一个食谱查询分析助手。分析用户的查询并返回结构化的 JSON 结果。

## 意图分类
- **recommendation**: 用户寻求菜品推荐（如"今天吃什么""推荐下饭菜"）
- **howto**: 询问烹饪步骤（如"怎么做红烧肉""麻婆豆腐步骤"）
- **ingredient**: 询问食材清单（如"红烧肉需要什么材料"）
- **factual**: 询问菜品知识（如"宫保鸡丁是什么菜系"）

## 约束提取规则
- difficulty: 简单/快手→★, 中等→★★, 困难/大厨→★★★★
- category: vegetable_dish, meat_dish, soup, breakfast, aquatic, staple, dessert
- calories: 低卡/减肥/清淡→low, 高热量/硬菜→high
- probes: 推荐意图时生成3-5个不同角度的搜索词"""


class LLMIntentClassifier:
    """Intent classifier using LangChain ChatOpenAI with structured output."""

    def __init__(
        self,
        model: str = LLM_INTENT_MODEL,
        api_base: str = LLM_INTENT_API_BASE,
        api_key: Optional[str] = None,
        fallback: bool = True,
    ):
        self.model = model
        self.api_base = api_base
        self.fallback = fallback

        api_key = api_key or os.environ.get(LLM_API_KEY_ENV)
        self._client = None
        self._structured_llm = None

        if api_key:
            llm = ChatOpenAI(
                model=model,
                base_url=api_base,
                api_key=api_key,
                temperature=LLM_INTENT_TEMPERATURE,
                timeout=LLM_INTENT_TIMEOUT,
            )
            self._client = llm
            self._structured_llm = llm.with_structured_output(IntentClassificationSchema)

    def classify(self, query: str) -> IntentResult:
        if not self._client or not self._structured_llm:
            return self._do_fallback(query)
        try:
            return self._llm_classify(query)
        except Exception:
            return self._do_fallback(query)

    def _llm_classify(self, query: str) -> IntentResult:
        result = self._structured_llm.invoke(
            f"{INTENT_SYSTEM_PROMPT}\n\n用户查询：{query}"
        )
        return self._to_intent_result(result, query)

    def _to_intent_result(self, parsed: IntentClassificationSchema, query: str) -> IntentResult:
        filters: Dict = {}
        if parsed.filters:
            if parsed.filters.difficulty:
                filters["difficulty"] = parsed.filters.difficulty
            if parsed.filters.category:
                filters["category"] = parsed.filters.category
            if parsed.filters.calories in ("low", "high"):
                filters["calories"] = parsed.filters.calories

        result = IntentResult(
            intent=parsed.intent,
            rewritten=parsed.rewritten_query or query,
            filters=filters,
            target_dish=parsed.target_dish,
            confidence=parsed.confidence,
        )

        if parsed.intent == IntentType.RECOMMENDATION:
            result.probes = parsed.probes or build_probes(query, filters)

        return result

    def _do_fallback(self, query: str) -> IntentResult:
        return classify_intent(query)
