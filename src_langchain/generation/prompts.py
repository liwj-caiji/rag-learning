"""Prompt templates for RAG response generation."""

from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


_BASE_SYSTEM = (
    "你是一个中文烹饪助手，根据检索到的食谱信息回答用户的问题。\n\n"
    "要求：\n"
    "1. 只使用提供的上下文信息回答，不要编造菜谱\n"
    "2. 如果上下文信息不足，请如实告知用户\n"
    "3. 回答要结构清晰、易于阅读\n"
    "4. 使用中文回答\n"
    "5. 可以在回答中引用菜名作为来源\n"
)

RECOMMENDATION_SYSTEM_PROMPT = _BASE_SYSTEM + "\n推荐场景：给出推荐理由，列出菜品名称、类别、难度和预估卡路里。"

HOWTO_SYSTEM_PROMPT = _BASE_SYSTEM + (
    "\n步骤说明：检查检索到的上下文中是否包含用户询问的菜名。"
    "\n- 上下文中可能包含标注为「完整食谱」的条目，"
    "这是该菜品的完整源文件，包含介绍、原料、步骤等全部信息，"
    "应优先使用此条目来回答。\n"
    "- 其他条目是检索到的相关片段，可作为补充参考。\n"
    "- 如果上下文中包含该菜品的操作步骤，则按顺序整理回答。\n"
    "- 如果上下文中没有该菜品的信息，请如实回答"
    "「抱歉，我不知道这道菜的做法」，"
    "然后根据检索到的其他菜品推荐相似的选择。\n"
    "- 不要强行将其他菜品的步骤套用到用户询问的菜品上。"
)

INGREDIENT_SYSTEM_PROMPT = _BASE_SYSTEM + (
    "\n原料清单：检查检索到的上下文中是否包含用户询问的菜名。"
    "\n- 如果上下文中包含该菜品的原料清单，则整理列出。\n"
    "- 如果上下文中没有该菜品的信息，请如实回答"
    "「抱歉，我不知道这道菜的原料信息」，"
    "然后根据检索到的其他菜品推荐相似的选择。\n"
    "- 不要强行将其他菜品的原料信息套用到用户询问的菜品上。"
)

FACTUAL_SYSTEM_PROMPT = _BASE_SYSTEM + "\n知识回答：根据上下文直接回答，如信息不足则说明。"


def get_system_prompt(intent: str, target_dish: Optional[str] = None) -> str:
    prompts = {
        "recommendation": RECOMMENDATION_SYSTEM_PROMPT,
        "howto": HOWTO_SYSTEM_PROMPT,
        "ingredient": INGREDIENT_SYSTEM_PROMPT,
        "factual": FACTUAL_SYSTEM_PROMPT,
    }
    prompt = prompts.get(intent, FACTUAL_SYSTEM_PROMPT)
    if target_dish and intent in ("howto", "ingredient"):
        label = "做法" if intent == "howto" else "原料信息"
        prompt += f"\n用户询问「{target_dish}」的{label}。"
    return prompt


def build_user_prompt(query: str, context_text: str) -> str:
    return f"""用户问题：{query}

检索到的食谱信息：
{context_text}

请根据以上信息回答用户的问题。"""


def format_context(documents: List[Document]) -> str:
    sections = []
    for i, doc in enumerate(documents, 1):
        meta = doc.metadata
        dish = meta.get("dish_name", "?")
        section_type = meta.get("section_type", "")
        subsection = meta.get("subsection_name", "")

        header = f"[{i}] {dish}"
        if section_type:
            header += f" - {section_type}"
        if subsection:
            header += f" ({subsection})"

        text = doc.page_content.strip()
        sections.append(f"{header}\n{text}\n")

    return "\n---\n".join(sections)
