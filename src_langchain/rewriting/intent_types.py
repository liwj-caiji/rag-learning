"""Intent classification types and shared rule-matching logic."""

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

_DIFFICULTY_PATTERNS = [
    (re.compile(r"简单|容易|快手|新手|入门"), "★"),
    (re.compile(r"中等|中级"), "★★"),
    (re.compile(r"困难|复杂|高级|大厨"), "★★★★"),
]

_CATEGORY_KEYWORDS = {
    "vegetable_dish": ["素菜", "蔬菜", "青菜", "凉拌"],
    "meat_dish": ["肉菜", "荤菜", "猪肉", "牛肉", "鸡肉", "排骨"],
    "soup": ["汤", "羹"],
    "breakfast": ["早餐", "早饭", "早点"],
    "aquatic": ["鱼", "虾", "蟹", "水产", "海鲜"],
    "staple": ["主食", "饭", "面", "粥"],
    "dessert": ["甜点", "甜品", "蛋糕", "糖水"],
}

_CALORIE_LOW_PATTERNS = re.compile(r"低热量|低卡|减肥|瘦身|清淡|低脂")
_CALORIE_HIGH_PATTERNS = re.compile(r"高热量|高卡|增肥|硬菜|横菜")

_DISH_PATTERN = re.compile(
    r"(?:怎么做|的做法|如何做|怎么烧|如何烧|怎么煮|如何煮|"
    r"需要什么材料|需要哪些材料|的材料|的原料|的调料)\s*$"
)


def match_intent(query: str) -> str:
    q = query.strip()
    matched_intent = IntentType.RECOMMENDATION
    matched_len = 0
    for intent, keywords in _INTENT_RULES:
        for kw in keywords:
            if kw in q and len(kw) > matched_len:
                matched_intent = intent
                matched_len = len(kw)
    return matched_intent


def extract_filters(query: str) -> Dict:
    q = query.strip()
    filters: Dict = {}
    for pattern, level in _DIFFICULTY_PATTERNS:
        if pattern.search(q):
            filters["difficulty"] = level
            break
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                filters["category"] = cat
                break
        if "category" in filters:
            break
    if _CALORIE_LOW_PATTERNS.search(q):
        filters["calories"] = "low"
    elif _CALORIE_HIGH_PATTERNS.search(q):
        filters["calories"] = "high"
    return filters


def extract_target_dish(query: str, intent: str) -> Optional[str]:
    if intent not in (IntentType.HOWTO, IntentType.INGREDIENT):
        return None
    rest = _DISH_PATTERN.sub("", query).strip()
    return rest if rest else None


def build_rewritten_query(query: str, intent: str, target_dish: Optional[str], filters: Dict) -> str:
    if intent == IntentType.RECOMMENDATION:
        stripped = re.sub(r"今天|吃什么|吃啥|推荐|来点|想|有没有|给我", "", query).strip()
        return stripped if stripped else "家常菜"
    if intent == IntentType.HOWTO and target_dish:
        return f"{target_dish} 操作 步骤"
    if intent == IntentType.INGREDIENT and target_dish:
        return f"{target_dish} 必备原料和工具"
    return query


def build_probes(query: str, filters: Dict) -> List[str]:
    base = build_rewritten_query(query, IntentType.RECOMMENDATION, None, filters)
    probes = [base]
    cat = filters.get("category")
    if cat:
        probes.append(f"{cat} {base}")
    else:
        probes.append(f"家常菜 {base}")
        probes.append(f"下饭 {base}")
    if filters.get("difficulty") == "★":
        probes.append(f"简单快手 {base}")
    return probes[:5]
