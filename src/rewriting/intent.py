"""意图分类 + 约束提取（规则驱动）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class IntentType:
    RECOMMENDATION = "recommendation"
    HOWTO = "howto"
    INGREDIENT = "ingredient"
    FACTUAL = "factual"


@dataclass
class IntentResult:
    intent: str
    rewritten: str = ""
    probes: List[str] = field(default_factory=list)
    filters: Dict = field(default_factory=dict)
    target_dish: Optional[str] = None
    confidence: float = 1.0


# ---- Intent keyword rules ----

_INTENT_RULES = [
    (IntentType.RECOMMENDATION, [
        "今天吃什么", "吃啥", "吃点啥", "推荐菜", "推荐",
        "下饭菜", "来点", "适合", "有什么", "好吃的",
        "清淡的", "开胃", "想",
    ]),
    (IntentType.HOWTO, [
        "怎么做", "做法", "步骤", "如何做", "操作",
        "怎么烧", "如何烧", "怎么煮", "如何煮",
    ]),
    (IntentType.INGREDIENT, [
        "材料", "原料", "食材", "调料", "配料",
        "需要什么", "清单", "准备",
    ]),
    (IntentType.FACTUAL, [
        "是什么", "哪里的菜", "起源于", "由来", "来历",
        "区别", "是什么菜",
    ]),
]

# ---- Constraint extraction patterns ----

_DIFFICULTY_PATTERNS = [
    (re.compile(r"简单|容易|快手|新手|入门"), "★"),
    (re.compile(r"中等|中级"), "★★"),
    (re.compile(r"困难|复杂|高级|大厨"), "★★★★"),
]

_CATEGORY_KEYWORDS = {
    "素菜": ["素菜", "蔬菜", "青菜", "凉拌"],
    "肉菜": ["肉菜", "荤菜", "猪肉", "牛肉", "鸡肉", "排骨"],
    "汤": ["汤", "羹"],
    "早餐": ["早餐", "早饭", "早点"],
    "水产": ["鱼", "虾", "蟹", "水产", "海鲜"],
    "主食": ["主食", "饭", "面", "粥"],
    "甜点": ["甜点", "甜品", "蛋糕", "糖水"],
}

_CALORIE_LOW_PATTERNS = re.compile(r"低热量|低卡|减肥|瘦身|清淡|低脂")
_CALORIE_HIGH_PATTERNS = re.compile(r"高热量|高卡|增肥|硬菜|横菜")

# ---- Dish name extraction for howto/ingredient ----

_DISH_PATTERN = re.compile(
    r"(?:怎么做|的做法|如何做|怎么烧|如何烧|怎么煮|如何煮|"
    r"需要什么材料|需要哪些材料|的材料|的原料|的调料)\s*$"
)


def classify_intent(query: str) -> IntentResult:
    """Classify query intent and extract constraints.

    Pure rule-based, no external dependencies.
    """
    q = query.strip()

    # 1. Determine intent by keyword matching (longest match wins)
    matched_intent = IntentType.RECOMMENDATION
    matched_len = 0
    for intent, keywords in _INTENT_RULES:
        for kw in keywords:
            if kw in q and len(kw) > matched_len:
                matched_intent = intent
                matched_len = len(kw)

    # 2. Extract filters (difficulty, category, calories)
    filters: Dict = {}

    # Difficulty
    for pattern, level in _DIFFICULTY_PATTERNS:
        if pattern.search(q):
            filters["difficulty"] = level
            break

    # Category
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                filters["category"] = cat
                break
        if "category" in filters:
            break

    # Calories
    if _CALORIE_LOW_PATTERNS.search(q):
        filters["calories"] = "low"
    elif _CALORIE_HIGH_PATTERNS.search(q):
        filters["calories"] = "high"

    # 3. Extract target dish (for howto / ingredient)
    target_dish = None
    if matched_intent in (IntentType.HOWTO, IntentType.INGREDIENT):
        # Remove intent suffix to get dish name
        rest = _DISH_PATTERN.sub("", q).strip()
        if rest:
            target_dish = rest

    # 4. Build result
    result = IntentResult(
        intent=matched_intent,
        rewritten=_build_rewritten_query(q, matched_intent, target_dish, filters),
        filters=filters,
        target_dish=target_dish,
    )

    # 5. Generate multiple search probes (for recommendation diversity)
    if matched_intent == IntentType.RECOMMENDATION:
        result.probes = _build_probes(q, filters)

    return result


def _build_rewritten_query(
    query: str,
    intent: str,
    target_dish: Optional[str],
    filters: Dict,
) -> str:
    """Build an effective search query from the original query."""
    if intent == IntentType.RECOMMENDATION:
        # Strip recommendation keywords, keep food-related terms
        stripped = re.sub(
            r"今天|吃什么|吃啥|推荐|来点|想|有没有|给我",
            "", query
        ).strip()
        if not stripped:
            stripped = "家常菜"
        return stripped

    if intent == IntentType.HOWTO and target_dish:
        return f"{target_dish} 操作 步骤"

    if intent == IntentType.INGREDIENT and target_dish:
        return f"{target_dish} 必备原料和工具"

    return query


def _build_probes(query: str, filters: Dict) -> List[str]:
    """Generate multiple search probes for diverse recommendation retrieval."""
    probes = []
    base = _build_rewritten_query(query, IntentType.RECOMMENDATION, None, filters)

    # Main probe
    probes.append(base)

    # Category-specific probes
    cat = filters.get("category")
    if cat:
        probes.append(f"{cat} {base}")
    else:
        probes.append(f"家常菜 {base}")
        probes.append(f"下饭 {base}")

    # Difficulty-based
    diff = filters.get("difficulty")
    if diff == "★":
        probes.append(f"简单快手 {base}")

    return probes[:5]
