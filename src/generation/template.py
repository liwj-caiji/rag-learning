"""Template-based generator — no LLM needed, for testing pipeline."""

from __future__ import annotations

from typing import Dict, List, Optional

from .base import Generator


class TemplateGenerator(Generator):
    """Generate responses using simple f-string templates.

    Used for pipeline testing before integrating an LLM.
    """

    @staticmethod
    def _get_chunk(item: Dict) -> Dict:
        """Extract the inner chunk, handling both raw chunks and wrapped results."""
        return item.get("chunk", item)

    @staticmethod
    def _dishes_in_context(context: List[Dict]) -> set:
        """Return the set of dish names present in the retrieved context."""
        dishes = set()
        for item in context:
            chunk = item.get("chunk", item)
            dish = chunk.get("metadata", {}).get("dish_name", "")
            if dish:
                dishes.add(dish)
        return dishes

    def generate(
        self,
        query: str,
        context: List[Dict],
        intent: str,
        target_dish: Optional[str] = None,
    ) -> str:
        if intent == "recommendation":
            return self._render_recommendation(context)
        elif intent == "howto":
            return self._render_howto(context, target_dish=target_dish)
        elif intent == "ingredient":
            return self._render_ingredient(context, target_dish=target_dish)
        else:
            return self._render_generic(context)

    def _render_recommendation(self, context: List[Dict]) -> str:
        if not context:
            return "抱歉，没有找到符合条件的菜品推荐。"

        lines = ["为您推荐以下菜品：\n"]
        for i, item in enumerate(context, 1):
            chunk = self._get_chunk(item)
            meta = chunk.get("metadata", {})
            dish = meta.get("dish_name", "?")
            cat = meta.get("category", "?")
            diff = meta.get("difficulty", "?")
            cal = meta.get("calories", "?")
            lines.append(
                f"{i}. 【{dish}】\n"
                f"   类别：{cat}  |  难度：{diff}  |  卡路里：{cal}\n"
            )
        return "\n".join(lines)

    def _render_howto(self, context: List[Dict], target_dish: Optional[str] = None) -> str:
        if not context:
            if target_dish:
                return f"抱歉，我不知道「{target_dish}」的做法。"
            return "抱歉，没有找到该菜品的做法。"

        # Check if target dish was actually found in context
        actual_dishes = self._dishes_in_context(context)
        if target_dish and target_dish not in actual_dishes:
            recs = "、".join(sorted(actual_dishes)[:5])
            if recs:
                return (
                    f"抱歉，我不知道「{target_dish}」的做法。"
                    f"为您推荐以下相似的菜品：{recs}"
                )
            return f"抱歉，我不知道「{target_dish}」的做法。"

        # Group by dish name
        by_dish: Dict[str, List[str]] = {}
        for item in context:
            chunk = self._get_chunk(item)
            meta = chunk.get("metadata", {})
            dish = meta.get("dish_name", "?")
            text = chunk.get("text", "")
            by_dish.setdefault(dish, []).append(text)

        lines = []
        for dish, texts in by_dish.items():
            lines.append(f"【{dish}的做法】\n")
            for t in texts:
                lines.append(t)
            lines.append("")

        return "\n".join(lines)

    def _render_ingredient(self, context: List[Dict], target_dish: Optional[str] = None) -> str:
        if not context:
            if target_dish:
                return f"抱歉，我不知道「{target_dish}」的原料信息。"
            return "抱歉，没有找到该菜品的原料信息。"

        # Check if target dish was actually found in context
        actual_dishes = self._dishes_in_context(context)
        if target_dish and target_dish not in actual_dishes:
            recs = "、".join(sorted(actual_dishes)[:5])
            if recs:
                return (
                    f"抱歉，我不知道「{target_dish}」的原料信息。"
                    f"为您推荐以下相似的菜品：{recs}"
                )
            return f"抱歉，我不知道「{target_dish}」的原料信息。"

        lines = []
        for item in context:
            chunk = self._get_chunk(item)
            meta = chunk.get("metadata", {})
            dish = meta.get("dish_name", "?")
            text = chunk.get("text", "")
            lines.append(f"【{dish}】所需材料：\n{text}\n")

        return "\n".join(lines)

    def _render_generic(self, context: List[Dict]) -> str:
        if not context:
            return "抱歉，没有找到相关信息。"

        lines = ["找到以下相关信息：\n"]
        for i, item in enumerate(context, 1):
            chunk = self._get_chunk(item)
            meta = chunk.get("metadata", {})
            dish = meta.get("dish_name", "?")
            text = chunk.get("text", "")
            preview = text[:150].replace("\n", " ")
            lines.append(f"{i}. [{dish}] {preview}...\n")
        return "\n".join(lines)
