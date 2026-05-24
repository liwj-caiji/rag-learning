"""Metadata-based chunk filtering."""

from typing import Dict, List, Optional


def apply_filters(
    chunks: List[Dict],
    filters: Optional[Dict] = None,
) -> List[Dict]:
    """Filter chunks by metadata fields.

    Supported filters:
      - difficulty: str (e.g. "★", "★★") — prefix match on chunk difficulty
      - category: str (e.g. "素菜", "肉菜") — exact match
      - calories: "low" or "high" — numeric comparison on kcal value
      - level: str (e.g. "dish", "section") — chunk level filter
      - target_dish: str — dish_name substring match
    """
    if not filters:
        return list(chunks)

    result = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})

        if not _match_difficulty(meta, filters):
            continue
        if not _match_category(meta, filters):
            continue
        if not _match_calories(meta, filters):
            continue
        if not _match_level(chunk, filters):
            continue
        if not _match_dish(meta, filters):
            continue

        result.append(chunk)

    return result


def _match_difficulty(meta: Dict, filters: Dict) -> bool:
    diff = filters.get("difficulty")
    if not diff:
        return True
    meta_diff = meta.get("difficulty", "")
    if not meta_diff:
        return True  # pass through if no difficulty info
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
    # Extract numeric value
    import re
    m = re.search(r"(\d+)", cal_str)
    if not m:
        return True
    kcal = int(m.group(1))
    if cal == "low":
        return kcal <= 300
    elif cal == "high":
        return kcal >= 600
    return True


def _match_level(chunk: Dict, filters: Dict) -> bool:
    lvl = filters.get("level")
    if not lvl:
        return True
    return chunk.get("level", "") == lvl


def _match_dish(meta: Dict, filters: Dict) -> bool:
    dish = filters.get("target_dish")
    if not dish:
        return True
    return dish.lower() in meta.get("dish_name", "").lower()
