"""Metadata-based document filtering."""

import re
from typing import Dict, List

from langchain_core.documents import Document

from ..config import CALORIE_LOW_THRESHOLD, CALORIE_HIGH_THRESHOLD


def apply_filters(
    documents: List[Document],
    filters: Dict,
) -> List[Document]:
    if not filters:
        return list(documents)

    result = []
    for doc in documents:
        meta = doc.metadata
        if not _match_difficulty(meta, filters):
            continue
        if not _match_category(meta, filters):
            continue
        if not _match_calories(meta, filters):
            continue
        if not _match_level(meta, filters):
            continue
        if not _match_dish(meta, filters):
            continue
        result.append(doc)

    return result


def _match_difficulty(meta: Dict, filters: Dict) -> bool:
    diff = filters.get("difficulty")
    if not diff:
        return True
    meta_diff = meta.get("difficulty", "")
    if not meta_diff:
        return True
    return meta_diff.startswith(diff)


def _match_category(meta: Dict, filters: Dict) -> bool:
    cat = filters.get("category")
    if not cat:
        return True
    return meta.get("category", "") == cat


def _match_calories(meta: Dict, filters: Dict) -> bool:
    cal = filters.get("calories")
    if not cal:
        return True
    cal_str = meta.get("calories", "")
    if not cal_str:
        return True
    m = re.search(r"(\d+)", cal_str)
    if not m:
        return True
    kcal = int(m.group(1))
    if cal == "low":
        return kcal <= CALORIE_LOW_THRESHOLD
    elif cal == "high":
        return kcal >= CALORIE_HIGH_THRESHOLD
    return True


def _match_level(meta: Dict, filters: Dict) -> bool:
    lvl = filters.get("level")
    if not lvl:
        return True
    return meta.get("level", "") == lvl


def _match_dish(meta: Dict, filters: Dict) -> bool:
    dish = filters.get("target_dish")
    if not dish:
        return True
    return dish.lower() in meta.get("dish_name", "").lower()
