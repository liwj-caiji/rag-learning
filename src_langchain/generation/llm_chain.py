"""LLM-powered response generator using LangChain ChatOpenAI."""

import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..config import (
    LLM_GEN_MODEL, LLM_GEN_API_BASE, LLM_GEN_TIMEOUT,
    LLM_GEN_TEMPERATURE, LLM_GEN_MAX_TOKENS, LLM_API_KEY_ENV,
)
from .base import Generator
from .prompts import get_system_prompt, build_user_prompt, format_context


class LLMGenerator(Generator):
    """Generate responses using ChatOpenAI + retrieved context."""

    def __init__(
        self,
        model: str = LLM_GEN_MODEL,
        api_base: str = LLM_GEN_API_BASE,
        api_key: Optional[str] = None,
        fallback: bool = True,
        callbacks: Optional[List] = None,
    ):
        self.model = model
        self.api_base = api_base
        self.fallback = fallback
        self.callbacks = callbacks

        api_key = api_key or os.environ.get(LLM_API_KEY_ENV)
        self._llm = None
        if api_key:
            self._llm = ChatOpenAI(
                model=model,
                base_url=api_base,
                api_key=api_key,
                temperature=LLM_GEN_TEMPERATURE,
                max_tokens=LLM_GEN_MAX_TOKENS,
                timeout=LLM_GEN_TIMEOUT,
            )

        if self.fallback:
            from .template_chain import TemplateGenerator
            self._fallback_gen = TemplateGenerator()
        else:
            self._fallback_gen = None

    def generate(
        self,
        query: str,
        context: List[Document],
        intent: str,
        target_dish: Optional[str] = None,
    ) -> str:
        if not self._llm:
            return (
                self._fallback_gen.generate(query, context, intent, target_dish=target_dish)
                if self._fallback_gen else ""
            )

        if not context:
            return self._empty_response(intent, target_dish=target_dish)

        try:
            return self._llm_generate(query, context, intent, target_dish=target_dish)
        except Exception:
            if self._fallback_gen:
                return self._fallback_gen.generate(query, context, intent, target_dish=target_dish)
            return ""

    def _llm_generate(
        self,
        query: str,
        context: List[Document],
        intent: str,
        target_dish: Optional[str] = None,
    ) -> str:
        system_prompt = get_system_prompt(intent, target_dish=target_dish)
        context_text = format_context(context)
        user_prompt = build_user_prompt(query, context_text)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{user_prompt}"),
        ])

        chain = prompt | self._llm | StrOutputParser()
        invoke_config = {}
        if self.callbacks:
            invoke_config["callbacks"] = self.callbacks
        return chain.invoke({"user_prompt": user_prompt}, config=invoke_config)

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
