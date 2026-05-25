"""LLM-powered intent classification with structured output."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from openai import OpenAI

from src.config import (
    LLM_INTENT_MODEL, LLM_INTENT_API_BASE, LLM_INTENT_TIMEOUT,
    LLM_INTENT_TEMPERATURE, LLM_INTENT_MAX_TOKENS, LLM_API_KEY_ENV,
)
from .intent import IntentResult, IntentType, _build_rewritten_query, _build_probes


class LLMIntentClassifier:
    """Intent classifier powered by LLM with structured JSON output.

    Uses OpenAI-compatible API (DeepSeek). Falls back to rule-based
    classification on API error or missing API key.

    Usage:
        classifier = LLMIntentClassifier()
        result = classifier.classify("今天想吃点清淡的")
        print(result.intent, result.filters)
    """

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
        if api_key:
            self._client = OpenAI(
                api_key=api_key, base_url=api_base, timeout=LLM_INTENT_TIMEOUT,
            )
        else:
            self._client = None

    def classify(self, query: str) -> IntentResult:
        """Classify intent using LLM, fall back to rules if needed."""
        if not self._client:
            return self._fallback(query)

        try:
            return self._llm_classify(query)
        except Exception:
            return self._fallback(query)

    def _llm_classify(self, query: str) -> IntentResult:
        """Call LLM with structured output for intent classification."""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"},
            temperature=LLM_INTENT_TEMPERATURE,
            max_tokens=LLM_INTENT_MAX_TOKENS,
            timeout=LLM_INTENT_TIMEOUT,
        )

        raw = resp.choices[0].message.content
        if not raw:
            raise ValueError("Empty LLM response")

        data = json.loads(raw)
        return self._parse_result(data, query)

    def _parse_result(self, data: Dict, query: str) -> IntentResult:
        """Parse structured LLM output into IntentResult."""
        intent = data.get("intent", IntentType.RECOMMENDATION)

        filters: Dict = {}
        raw_filters = data.get("filters", {}) or {}
        if raw_filters.get("difficulty"):
            filters["difficulty"] = str(raw_filters["difficulty"])
        if raw_filters.get("category"):
            filters["category"] = str(raw_filters["category"])
        if raw_filters.get("calories") in ("low", "high"):
            filters["calories"] = raw_filters["calories"]

        target_dish = data.get("target_dish") or None
        rewritten = data.get("rewritten_query") or query

        # Build result matching rule-based interface
        result = IntentResult(
            intent=intent,
            rewritten=rewritten,
            filters=filters,
            target_dish=target_dish,
            confidence=data.get("confidence", 0.8),
        )

        # Generate probes for recommendation intent
        if intent == IntentType.RECOMMENDATION:
            result.probes = data.get("probes") or _build_probes(query, filters)

        return result

    def _fallback(self, query: str) -> IntentResult:
        """Fall back to rule-based classification."""
        from .intent import classify_intent
        return classify_intent(query)

    @classmethod
    def rewrite(cls, query: str) -> IntentResult:
        """Convenience one-shot method."""
        return cls().classify(query)


# ---- Prompt ----

_SYSTEM_PROMPT = """你是一个食谱查询分析助手。分析用户的查询并输出 JSON。

## 意图分类
- **recommendation**: 用户寻求菜品推荐（如"今天吃什么""推荐下饭菜"）
- **howto**: 询问烹饪步骤（如"怎么做红烧肉""麻婆豆腐步骤"）
- **ingredient**: 询问食材清单（如"红烧肉需要什么材料"）
- **factual**: 询问菜品知识（如"宫保鸡丁是什么菜系"）

## 输出 JSON 结构
```json
{
  "intent": "recommendation|howto|ingredient|factual",
  "rewritten_query": "用于搜索的改写查询（去除语气词，保留食物关键词）",
  "filters": {
    "difficulty": null 或 "★" / "★★" / "★★★★",
    "category": null 或 "vegetable_dish" / "meat_dish" / "soup" / "breakfast" / "aquatic" / "staple" / "dessert",
    "calories": null 或 "low" / "high"
  },
  "target_dish": null 或 "具体的菜名（不要带"的做法"后缀）",
  "probes": ["推荐场景的多个搜索探针，3-5个"],
  "confidence": 0.0-1.0
}
```

## 约束提取规则
- difficulty: 简单/快手→★, 中等→★★, 困难/大厨→★★★★
- category: 按食材/类型推断（vegetable_dish, 猪肉/牛肉→meat_dish, 鱼虾→aquatic, 汤→soup, 早餐→breakfast, 主食→staple, 甜点→dessert）
- calories: 低卡/减肥/清淡→low, 高热量/硬菜→high
- probes: 推荐意图时生成3-5个不同角度的搜索词"""
