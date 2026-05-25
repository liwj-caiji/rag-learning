"""LLM-powered response generator using DeepSeek API."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from openai import OpenAI

from src.config import (
    LLM_GEN_MODEL, LLM_GEN_API_BASE, LLM_GEN_TIMEOUT,
    LLM_GEN_TEMPERATURE, LLM_GEN_MAX_TOKENS, LLM_API_KEY_ENV,
)
from .base import Generator


class LLMGenerator(Generator):
    """Generate natural language responses using LLM + retrieved context.

    Usage:
        generator = LLMGenerator()
        answer = generator.generate("麻婆豆腐怎么做", context_list, "howto")
    """

    def __init__(
        self,
        model: str = LLM_GEN_MODEL,
        api_base: str = LLM_GEN_API_BASE,
        api_key: Optional[str] = None,
        fallback: bool = True,
    ):
        self.model = model
        self.api_base = api_base
        self.fallback = fallback

        api_key = api_key or os.environ.get(LLM_API_KEY_ENV)
        self._client = OpenAI(
            api_key=api_key, base_url=api_base, timeout=LLM_GEN_TIMEOUT,
        ) if api_key else None

        if self.fallback:
            from .template import TemplateGenerator
            self._fallback_gen = TemplateGenerator()
        else:
            self._fallback_gen = None

    def generate(self, query: str, context: List[Dict], intent: str, target_dish: Optional[str] = None) -> str:
        if not self._client:
            return self._fallback_gen.generate(query, context, intent, target_dish=target_dish) if self._fallback_gen else ""

        if not context:
            return self._empty_response(intent, target_dish=target_dish)

        try:
            return self._llm_generate(query, context, intent, target_dish=target_dish)
        except Exception:
            if self._fallback_gen:
                return self._fallback_gen.generate(query, context, intent, target_dish=target_dish)
            return ""

    def _llm_generate(self, query: str, context: List[Dict], intent: str, target_dish: Optional[str] = None) -> str:
        system_prompt = self._build_system_prompt(intent, target_dish=target_dish)
        context_text = self._format_context(context, intent)
        user_prompt = self._build_user_prompt(query, context_text, intent)

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=LLM_GEN_TEMPERATURE,
            max_tokens=LLM_GEN_MAX_TOKENS,
            timeout=LLM_GEN_TIMEOUT,
        )

        return resp.choices[0].message.content or ""

    def _build_system_prompt(self, intent: str, target_dish: Optional[str] = None) -> str:
        base = "你是一个中文烹饪助手，根据检索到的食谱信息回答用户的问题。\n\n"
        base += "要求：\n"
        base += "1. 只使用提供的上下文信息回答，不要编造菜谱\n"
        base += "2. 如果上下文信息不足，请如实告知用户\n"
        base += "3. 回答要结构清晰、易于阅读\n"
        base += "4. 使用中文回答\n"
        base += "5. 可以在回答中引用菜名作为来源\n"

        if intent == "recommendation":
            base += "\n推荐场景：给出推荐理由，列出菜品名称、类别、难度和预估卡路里。"
        elif intent == "howto":
            base += "\n步骤说明：检查检索到的上下文中是否包含用户询问的菜名。"
            if target_dish:
                base += f"\n用户询问「{target_dish}」的做法。"
            base += (
                "\n- 如果上下文中包含该菜品的操作步骤，则按顺序整理回答。\n"
                "- 如果上下文中没有该菜品的信息，请如实回答"
                "「抱歉，我不知道这道菜的做法」，"
                "然后根据检索到的其他菜品推荐相似的选择。\n"
                "- 不要强行将其他菜品的步骤套用到用户询问的菜品上。"
            )
        elif intent == "ingredient":
            base += "\n原料清单：检查检索到的上下文中是否包含用户询问的菜名。"
            if target_dish:
                base += f"\n用户询问「{target_dish}」的原料信息。"
            base += (
                "\n- 如果上下文中包含该菜品的原料清单，则整理列出。\n"
                "- 如果上下文中没有该菜品的信息，请如实回答"
                "「抱歉，我不知道这道菜的原料信息」，"
                "然后根据检索到的其他菜品推荐相似的选择。\n"
                "- 不要强行将其他菜品的原料信息套用到用户询问的菜品上。"
            )
        elif intent == "factual":
            base += "\n知识回答：根据上下文直接回答，如信息不足则说明。"

        return base

    def _build_user_prompt(self, query: str, context_text: str, intent: str) -> str:
        return f"""用户问题：{query}

检索到的食谱信息：
{context_text}

请根据以上信息回答用户的问题。"""

    def _format_context(self, context: List[Dict], intent: str) -> str:
        """Format retrieved chunks into readable text for the LLM."""
        sections = []
        for i, item in enumerate(context, 1):
            chunk = item.get("chunk", item)
            meta = chunk.get("metadata", {})
            dish = meta.get("dish_name", "?")
            section_type = meta.get("section_type", "")
            subsection = meta.get("subsection_name", "")
            text = chunk.get("text", "")

            header = f"[{i}] {dish}"
            if section_type:
                header += f" - {section_type}"
            if subsection:
                header += f" ({subsection})"

            # Clean and truncate text
            cleaned = text.strip()
            sections.append(f"{header}\n{cleaned}\n")

        return "\n---\n".join(sections)

    @staticmethod
    def _empty_response(intent: str, target_dish: Optional[str] = None) -> str:
        if target_dish:
            if intent == "howto":
                return f"抱歉，我不知道「{target_dish}」的做法。"
            if intent == "ingredient":
                return f"抱歉，我不知道「{target_dish}」的原料信息。"
        messages = {
            "recommendation": "抱歉，没有找到符合条件的菜品推荐。",
            "howto": "抱歉，没有找到该菜品的做法信息。",
            "ingredient": "抱歉，没有找到该菜品的原料信息。",
            "factual": "抱歉，没有找到相关信息。",
        }
        return messages.get(intent, "抱歉，没有找到相关信息。")
