"""
食谱 RAG 系统 — Gradio Web UI

Usage:
    python app.py
"""

from __future__ import annotations

import os
import pickle
from collections import Counter
from typing import Dict, List, Optional, Tuple

import gradio as gr

# ---------------------------------------------------------------------------
# Lazy backend
# ---------------------------------------------------------------------------

_PIPELINE_CACHE: Dict[str, object] = {}


def _get_pipeline(use_llm: bool = False):
    key = f"pipe_{use_llm}"
    if key not in _PIPELINE_CACHE:
        from src.generation import RAGPipeline
        _PIPELINE_CACHE[key] = RAGPipeline(use_llm=use_llm)
    return _PIPELINE_CACHE[key]


def _run_pipeline(query: str, top_k: int, use_llm: bool) -> dict:
    """Run full pipeline and return trace."""
    if not query.strip():
        return {"intent": "", "rewritten": "", "filters": {}, "probes": [],
                "target_dish": None, "num_chunks": 0, "chunks": [], "answer": "请输入查询"}
    try:
        pipe = _get_pipeline(use_llm)
        return pipe.trace(query, top_k=top_k)
    except Exception as e:
        return {"intent": "error", "rewritten": "", "filters": {}, "probes": [],
                "target_dish": None, "num_chunks": 0, "chunks": [],
                "answer": f"❌ 执行失败：{e}"}


def _chat_answer(message: str, history: List, use_llm: bool, top_k: int) -> Tuple[List, str]:
    if not message or not message.strip():
        return history, ""
    try:
        pipe = _get_pipeline(use_llm)
        answer = pipe.run(message, top_k=top_k)
        history.append((message, answer))
    except Exception as e:
        history.append((message, f"❌ 错误：{e}"))
    return history, ""


def _search(query: str, k: int, mode: str) -> Tuple[str, List, List]:
    if not query.strip():
        return "请输入查询", [], []

    try:
        if "混合" in mode:
            from src.retrieval import hybrid_search
            results = hybrid_search(query, k=k)
        elif "稠密" in mode:
            from src.retrieval import dense_search
            results = dense_search(query, k=k)
        else:
            from src.retrieval import sparse_search
            results = sparse_search(query, k=k)
    except Exception as e:
        return f"❌ 检索失败：{e}", [], []

    rows = []
    for r in results:
        chunk = r["chunk"]
        meta = chunk.get("metadata", {})
        rows.append([
            round(r.get("score", 0), 4),
            meta.get("dish_name", "?"),
            meta.get("category", "?"),
            chunk.get("level", ""),
            chunk.get("text", "")[:120] + "...",
        ])
    return f"✅ 共 {len(results)} 条结果", rows, [f"检索模式：{mode}"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _load_stats() -> str:
    try:
        from src.preprocess.splitter import collect_all_recipes
        from src.preprocess.config import DISHES_DIR, VECTORSTORE_DIR

        recipes = collect_all_recipes()
        total = len(recipes)
        cats = Counter()
        for r in recipes:
            rel = os.path.relpath(r, DISHES_DIR)
            cat = rel.replace("\\", "/").split("/")[0]
            cats[cat] += 1

        cat_lines = "".join(f"- **{k}**：{v} 道\n" for k, v in sorted(cats.items()))

        chunks_path = os.path.join(VECTORSTORE_DIR, "chunks.pkl")
        if os.path.exists(chunks_path):
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            total_chunks = len(chunks)
            levels = Counter(c.get("level", "?") for c in chunks)
            level_lines = "".join(f"- **{k}**：{v} 个\n" for k, v in sorted(levels.items()))
        else:
            total_chunks = 0
            level_lines = "（未构建索引）\n"

        return f"""## 数据集统计

| 指标 | 数值 |
|------|------|
| 食谱总数 | {total} |
| 品类数 | {len(cats)} |
| 分块总数 | {total_chunks} |
| 向量维度 | 768 |
| 检索方式 | 混合检索（FAISS + BM25 + RRF） |

### 品类分布
{cat_lines}
### 分块层级分布
{level_lines}
"""
    except Exception as e:
        return f"❌ 加载失败：`{e}`"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
#header { text-align: center; margin-bottom: 0.5em; }
#chat-col { border-left: 1px solid var(--border-color-primary); padding-left: 1em !important; }
footer { display: none !important; }
"""

# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="amber",
    font=gr.themes.GoogleFont("Noto Sans SC"),
)

with gr.Blocks(title="🍳 食谱 RAG 系统", fill_width=True) as demo:

    # ==== Header ====
    gr.Markdown(
        "# 🍳 食谱 RAG 系统\n"
        "基于混合检索与检索增强生成的智能食谱问答系统 — "
        "支持规则 / LLM 双模式切换",
        elem_id="header",
    )

    # ==== Global state ====
    llm_state = gr.State(False)
    top_k_state = gr.State(5)

    # ==== Main ====
    with gr.Row():
        # -- LEFT: Tabs --
        with gr.Column(scale=7):
            with gr.Tabs():

                # ---- Tab 1: Pipeline ----
                with gr.Tab("📊 Pipeline 总览"):
                    with gr.Row():
                        pipe_query = gr.Textbox(
                            label="查询", scale=4,
                            placeholder="例：今天吃什么、麻婆豆腐怎么做、清淡的素菜推荐",
                        )
                        pipe_top_k = gr.Slider(1, 10, 5, step=1, label="返回数量", scale=1)
                    with gr.Row():
                        pipe_llm = gr.Checkbox(label="使用 LLM（需配置 DEEPSEEK_API_KEY）", value=False)
                        pipe_btn = gr.Button("▶ 执行 Pipeline", variant="primary", scale=1, size="lg")

                    with gr.Accordion("① 查询改写 / 意图识别", open=False):
                        pipe_intent = gr.JSON(label="意图分析结果")

                    with gr.Accordion("② 检索结果", open=False):
                        pipe_results = gr.Dataframe(
                            headers=["菜名", "层级", "章节", "来源文件"],
                            label="命中分块",
                        )

                    with gr.Accordion("③ 生成回答", open=True):
                        pipe_answer = gr.Markdown("点击 **执行 Pipeline** 查看结果")

                # ---- Tab 2: Search ----
                with gr.Tab("🔍 检索演示"):
                    gr.Markdown("单独测试 **稠密检索 / 稀疏检索 / 混合检索** 的效果对比")
                    with gr.Row():
                        srch_query = gr.Textbox(label="查询", scale=3, placeholder="输入搜索关键词")
                        srch_k = gr.Slider(1, 20, 10, step=1, label="返回数量", scale=1)
                    srch_mode = gr.Radio(
                        ["混合检索 (Hybrid)", "稠密检索 (Dense)", "稀疏检索 (Sparse)"],
                        label="检索方式", value="混合检索 (Hybrid)",
                    )
                    srch_btn = gr.Button("🔎 检索", variant="primary")
                    srch_status = gr.Markdown("")
                    srch_results = gr.Dataframe(
                        headers=["得分", "菜名", "类别", "层级", "内容预览"],
                        label="检索结果",
                        wrap=True,
                    )

                # ---- Tab 3: Data ----
                with gr.Tab("📦 数据概览"):
                    data_stats = gr.Markdown("加载中…")
                    gr.Markdown("---")
                    gr.Markdown("**提示**：如需更新统计，请先运行 `python -m src.preprocess.indexer` 重建索引后点击刷新。")
                    data_refresh = gr.Button("🔄 刷新")

                # ---- Tab 4: About ----
                with gr.Tab("📖 关于"):
                    gr.Markdown("""
## 食谱 RAG 系统 v1.1

### 技术架构

| 模块 | 实现 |
|------|------|
| **分块** | 层级 Markdown 分块（dish / section / subsection） |
| **稠密检索** | FAISS + `shibing624/text2vec-base-chinese`（768 维） |
| **稀疏检索** | BM25 + jieba 中文分词 |
| **融合策略** | RRF（Reciprocal Rank Fusion, k=60） |
| **意图识别** | 规则匹配 / LLM（deepseek-v4-flash） |
| **答案生成** | 模板渲染 / LLM（deepseek-v4-flash） |
| **重排序** | MMR + 类别轮询 |

### 数据源
[HowToCook](https://github.com/Anduin2017/HowToCook) 开源食谱库

### 项目地址
[github.com/liwj-caiji/rag-learning](https://github.com/liwj-caiji/rag-learning)
                    """)

        # -- RIGHT: Chatbot --
        with gr.Column(scale=3, elem_id="chat-col"):
            gr.Markdown("### 💬 对话助手")
            chatbot = gr.Chatbot(
                label="对话历史",
                height=480,
                avatar_images=(None, "🧑‍🍳"),
                autoscroll=True,
            )

            chat_input = gr.Textbox(
                label="输入问题",
                placeholder="输入你的问题，按 Enter 发送…",
                container=False,
            )
            with gr.Row():
                chat_send = gr.Button("发送", variant="primary", scale=1)
                chat_clear = gr.Button("🗑 清除", scale=1, min_width=80)

            with gr.Row():
                chat_mode_hint = gr.Markdown("当前模式：**规则**")
                chat_mode_toggle = gr.Checkbox(label="LLM 模式", value=False)

    # =====================================================================
    # Events
    # =====================================================================

    # -- Pipeline --
    def on_pipe_run(q, k, llm):
        trace = _run_pipeline(q, k, llm)
        intent_data = {
            "意图": trace.get("intent", ""),
            "改写查询": trace.get("rewritten", ""),
            "过滤条件": trace.get("filters", {}),
            "目标菜名": trace.get("target_dish", ""),
            "搜索探针": trace.get("probes", []),
            "命中 chunk 数": trace.get("num_chunks", 0),
        }
        rows = [
            [c.get("dish", "?"), c.get("level", ""), c.get("section", ""), c.get("dish", "?")]
            for c in trace.get("chunks", [])
        ]
        answer = trace.get("answer", "（无结果）")
        return intent_data, gr.Dataframe(value=rows, headers=["菜名", "层级", "章节", "来源"]), answer

    pipe_btn.click(
        fn=on_pipe_run,
        inputs=[pipe_query, pipe_top_k, pipe_llm],
        outputs=[pipe_intent, pipe_results, pipe_answer],
    )

    # -- Search --
    def on_srch(q, k, mode):
        if not q.strip():
            return "请输入查询", gr.Dataframe(value=[])
        status, rows, _ = _search(q, k, mode)
        return status, gr.Dataframe(
            value=rows,
            headers=["得分", "菜名", "类别", "层级", "内容预览"],
            wrap=True,
        )

    srch_btn.click(
        fn=on_srch,
        inputs=[srch_query, srch_k, srch_mode],
        outputs=[srch_status, srch_results],
    )

    # -- Chat --
    def on_chat_msg(msg, history, llm, top_k):
        return _chat_answer(msg, history, llm, top_k)

    chat_send.click(
        fn=on_chat_msg,
        inputs=[chat_input, chatbot, llm_state, top_k_state],
        outputs=[chatbot, chat_input],
    )
    chat_input.submit(
        fn=on_chat_msg,
        inputs=[chat_input, chatbot, llm_state, top_k_state],
        outputs=[chatbot, chat_input],
    )
    chat_clear.click(fn=lambda: [], outputs=[chatbot])

    # -- Global state sync --
    def on_llm_toggle(v):
        return v, "当前模式：**LLM** 🧠" if v else "当前模式：**规则** 📋"

    pipe_llm.change(
        fn=on_llm_toggle,
        inputs=[pipe_llm],
        outputs=[llm_state, chat_mode_hint],
    )
    chat_mode_toggle.change(
        fn=on_llm_toggle,
        inputs=[chat_mode_toggle],
        outputs=[llm_state, chat_mode_hint],
    )
    # Keep both checkboxes in sync
    pipe_llm.change(fn=lambda v: v, inputs=[pipe_llm], outputs=[chat_mode_toggle])
    chat_mode_toggle.change(fn=lambda v: v, inputs=[chat_mode_toggle], outputs=[pipe_llm])

    pipe_top_k.change(fn=lambda v: v, inputs=[pipe_top_k], outputs=[top_k_state])

    # -- Data stats --
    data_refresh.click(fn=_load_stats, outputs=[data_stats])
    demo.load(fn=_load_stats, outputs=[data_stats])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    share = "--share" in sys.argv
    demo.launch(
        share=share,
        server_name="127.0.0.1", server_port=7860,
        theme=theme, css=CUSTOM_CSS,
    )
