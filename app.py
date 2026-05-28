"""
Recipe RAG System — Gradio Web UI

Usage:
    python app.py                        # default: src backend
    python app.py --backend langchain    # LangChain backend
    python app.py --backend langchain --share
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import gradio as gr

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
_backend: str = "src"  # "src" | "langchain"


def _resolve(name: str):
    """Import a name from the current backend."""
    if _backend == "langchain":
        import importlib
        mod = importlib.import_module(f"src_langchain.{name}")
    else:
        import importlib
        mod = importlib.import_module(f"src.{name}")
    return mod


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _init_logging():
    if _backend == "langchain":
        from src_langchain.config import (LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT,
                                          LOG_DIR, LOG_FILE, LOG_SUPPRESS)
    else:
        from src.config import (LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT,
                                LOG_DIR, LOG_FILE, LOG_SUPPRESS)

    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    os.makedirs(LOG_DIR, exist_ok=True)
    _fh = logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a")
    _fh.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(_fh)

    for noisy in LOG_SUPPRESS:
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Lazy backend
# ---------------------------------------------------------------------------

_PIPELINE_CACHE: Dict[str, object] = {}


def _get_pipeline(use_llm: bool = False):
    key = f"pipe_{use_llm}"
    if key not in _PIPELINE_CACHE:
        log.info("Creating pipeline (use_llm=%s, backend=%s)", use_llm, _backend)
        t0 = time.time()
        mod = _resolve("pipeline")
        _PIPELINE_CACHE[key] = mod.RAGPipeline(use_llm=use_llm)
        log.info("Pipeline created in %.2fs", time.time() - t0)
    return _PIPELINE_CACHE[key]


def _warmup():
    log.info("Warmup: preloading model and index (backend=%s)...", _backend)
    t0 = time.time()
    try:
        if _backend == "langchain":
            from src_langchain.retrieval.ensemble import hybrid_search
        else:
            from src.retrieval.hybrid import hybrid_search
        hybrid_search("预热", k=1)
        log.info("Warmup done in %.2fs", time.time() - t0)
    except Exception as e:
        log.warning("Warmup failed (%s), first request may be slow", e)


def _run_pipeline(query: str, top_k: int, use_llm: bool) -> dict:
    if not query.strip():
        return {"intent": "", "rewritten": "", "filters": {}, "probes": [],
                "target_dish": None, "num_chunks": 0, "chunks": [], "answer": "请输入查询"}
    log.info("Pipeline start | query=%r top_k=%d use_llm=%s", query, top_k, use_llm)
    t0 = time.time()
    try:
        pipe = _get_pipeline(use_llm)
        result = pipe.trace(query, top_k=top_k)
        elapsed = time.time() - t0
        log.info("Pipeline done | intent=%s chunks=%d elapsed=%.2fs",
                 result.get("intent"), result.get("num_chunks"), elapsed)
        return result
    except Exception as e:
        elapsed = time.time() - t0
        log.error("Pipeline failed after %.2fs | %s: %s", elapsed, type(e).__name__, e)
        return {"intent": "error", "rewritten": "", "filters": {}, "probes": [],
                "target_dish": None, "num_chunks": 0, "chunks": [],
                "answer": f"执行失败：{e}"}


def _chat_answer(message: str, history: List, use_llm: bool, top_k: int) -> Tuple[List, str]:
    if not message or not message.strip():
        return history, ""
    log.info("Chat start | query=%r use_llm=%s top_k=%d", message, use_llm, top_k)
    t0 = time.time()
    try:
        pipe = _get_pipeline(use_llm)
        answer = pipe.run(message, top_k=top_k)
        elapsed = time.time() - t0
        log.info("Chat done | elapsed=%.2fs answer_len=%d", elapsed, len(answer))
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
    except Exception as e:
        elapsed = time.time() - t0
        log.error("Chat failed after %.2fs | %s: %s", elapsed, type(e).__name__, e)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"错误：{e}"})
    return history, ""


def _search(query: str, k: int, mode: str) -> Tuple[str, List, List]:
    if not query.strip():
        return "请输入查询", [], []

    log.info("Search start | query=%r k=%d mode=%s", query, k, mode)
    t0 = time.time()
    try:
        if _backend == "langchain":
            if "混合" in mode:
                from src_langchain.retrieval.ensemble import hybrid_search
                results = hybrid_search(query, k=k)
            elif "稠密" in mode:
                from src_langchain.retrieval.ensemble import _load_faiss
                faiss_store = _load_faiss()
                retriever = faiss_store.as_retriever(search_kwargs={"k": k})
                results = retriever.invoke(query)
            else:
                from src_langchain.retrieval.sparse_retriever import sparse_search
                results = sparse_search(query, k=k)
        else:
            if "混合" in mode:
                from src.retrieval import hybrid_search
                results = hybrid_search(query, k=k)
            elif "稠密" in mode:
                from src.retrieval import dense_search
                results = dense_search(query, k=k)
            else:
                from src.retrieval import sparse_search
                results = sparse_search(query, k=k)
        elapsed = time.time() - t0
        log.info("Search done | results=%d elapsed=%.2fs", len(results), elapsed)
    except Exception as e:
        elapsed = time.time() - t0
        log.error("Search failed after %.2fs | %s: %s", elapsed, type(e).__name__, e)
        return f"检索失败：{e}", [], []

    rows = []
    if _backend == "langchain":
        for doc in results:
            meta = doc.metadata
            rows.append([
                round(meta.get("rrf_score", meta.get("bm25_score", 0)), 4),
                meta.get("dish_name", "?"),
                meta.get("category", "?"),
                meta.get("level", ""),
                doc.page_content[:120] + "...",
            ])
    else:
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
    return f"共 {len(results)} 条结果", rows, [f"检索模式：{mode}"]


# ---------------------------------------------------------------------------
# Data stats
# ---------------------------------------------------------------------------

_STATS_CACHE: Optional[str] = None


def _load_stats() -> str:
    global _STATS_CACHE
    if _STATS_CACHE is not None:
        return _STATS_CACHE
    log.info("Loading data stats (backend=%s)...", _backend)
    t0 = time.time()
    try:
        if _backend == "langchain":
            from src_langchain.preprocess.loader import load_recipe_documents
            from src_langchain.config import DISHES_DIR, EMBED_DIM, VECTORSTORE_DIR
        else:
            from src.preprocess.splitter import collect_all_recipes
            from src.config import DISHES_DIR, EMBED_DIM, VECTORSTORE_DIR

        if _backend == "langchain":
            raw_docs = load_recipe_documents()
            total = len(raw_docs)
            cats = Counter()
            for doc in raw_docs:
                cats[doc.metadata.get("category", "unknown")] += 1
        else:
            recipes = collect_all_recipes()
            total = len(recipes)
            cats = Counter()
            for r in recipes:
                rel = os.path.relpath(r, DISHES_DIR)
                cats[rel.replace("\\", "/").split("/")[0]] += 1

        cat_lines = "".join(f"- **{k}**：{v} 道\n" for k, v in sorted(cats.items()))

        chunks_path = os.path.join(VECTORSTORE_DIR, "chunks.pkl")
        if os.path.exists(chunks_path):
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            total_chunks = len(chunks)
            if _backend == "langchain":
                levels = Counter(c.metadata.get("level", "?") for c in chunks)
            else:
                levels = Counter(c.get("level", "?") for c in chunks)
            level_lines = "".join(f"- **{k}**：{v} 个\n" for k, v in sorted(levels.items()))
        else:
            total_chunks = 0
            level_lines = "（未构建索引）\n"

        backend_label = " (LangChain)" if _backend == "langchain" else ""
        _STATS_CACHE = f"""## 数据集统计{backend_label}

| 指标 | 数值 |
|------|------|
| 食谱总数 | {total} |
| 品类数 | {len(cats)} |
| 分块总数 | {total_chunks} |
| 向量维度 | {EMBED_DIM} |
| 检索方式 | 混合检索（FAISS + BM25 + RRF） |

### 品类分布
{cat_lines}
### 分块层级分布
{level_lines}
"""
        log.info("Stats loaded in %.2fs", time.time() - t0)
        return _STATS_CACHE
    except Exception as e:
        return f"加载失败：`{e}`"


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

def _build_ui():
    backend_label = " (LangChain)" if _backend == "langchain" else ""
    title = f"食谱 RAG 系统{backend_label}"

    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="amber",
        font=gr.themes.GoogleFont("Noto Sans SC"),
    )

    about_md = f"""## 食谱 RAG 系统{" v2.0 (LangChain)" if _backend == "langchain" else " v1.x"}

### 技术架构

| 模块 | {"v2.0 (LangChain)" if _backend == "langchain" else "v1.x"} |
|------|------|
| **分块** | {"MarkdownHeaderTextSplitter + 自定义元数据提取" if _backend == "langchain" else "层级 Markdown 分块（dish / section / subsection）"} |
| **Embedding** | {"HuggingFaceEmbeddings" if _backend == "langchain" else "手动 SentenceTransformer"} |
| **稠密检索** | {"FAISS.from_documents()" if _backend == "langchain" else "FAISS + shibing624/text2vec-base-chinese（768 维）"} |
| **稀疏检索** | {"BM25Retriever + jieba" if _backend == "langchain" else "BM25 + jieba 中文分词"} |
| **融合策略** | {"RRFFusionRetriever(BaseRetriever)" if _backend == "langchain" else "RRF（Reciprocal Rank Fusion, k=60）"} |
| **意图识别** | {"ChatOpenAI.with_structured_output()" if _backend == "langchain" else "规则匹配 / LLM（deepseek-v4-flash）"} |
| **答案生成** | {"ChatOpenAI + ChatPromptTemplate + StrOutputParser" if _backend == "langchain" else "模板渲染 / LLM（deepseek-v4-flash）"} |

### 数据源
[HowToCook](https://github.com/Anduin2017/HowToCook) 开源食谱库

### 项目地址
[github.com/liwj-caiji/rag-learning](https://github.com/liwj-caiji/rag-learning)
"""

    index_cmd = "python -m src_langchain.preprocess.indexer" if _backend == "langchain" else "python -m src.preprocess.indexer"

    with gr.Blocks(title=title, fill_width=True) as demo:

        gr.Markdown(
            f"# {title}\n"
            "基于混合检索与检索增强生成的智能食谱问答系统 — "
            "支持规则 / LLM 双模式切换",
            elem_id="header",
        )

        llm_state = gr.State(False)
        top_k_state = gr.State(5)

        with gr.Row():
            with gr.Column(scale=7):
                with gr.Tabs():

                    with gr.Tab("Pipeline 总览"):
                        with gr.Row():
                            pipe_query = gr.Textbox(
                                label="查询", scale=4,
                                placeholder="例：今天吃什么、麻婆豆腐怎么做、清淡的素菜推荐",
                            )
                            pipe_top_k = gr.Slider(1, 10, 5, step=1, label="返回数量", scale=1)
                        with gr.Row():
                            pipe_llm = gr.Checkbox(label="使用 LLM（需配置 DEEPSEEK_API_KEY）", value=False)
                            pipe_btn = gr.Button("执行 Pipeline", variant="primary", scale=1, size="lg")

                        with gr.Accordion("① 查询改写 / 意图识别", open=False):
                            pipe_intent = gr.JSON(label="意图分析结果")

                        with gr.Accordion("② 检索结果", open=False):
                            pipe_results = gr.Dataframe(
                                headers=["菜名", "层级", "章节", "类别"],
                                label="命中分块",
                            )

                        with gr.Accordion("③ 生成回答", open=True):
                            pipe_answer = gr.Markdown("点击 **执行 Pipeline** 查看结果")

                    with gr.Tab("检索演示"):
                        gr.Markdown("单独测试 **稠密检索 / 稀疏检索 / 混合检索** 的效果对比")
                        with gr.Row():
                            srch_query = gr.Textbox(label="查询", scale=3, placeholder="输入搜索关键词")
                            srch_k = gr.Slider(1, 20, 10, step=1, label="返回数量", scale=1)
                        srch_mode = gr.Radio(
                            ["混合检索 (Hybrid)", "稠密检索 (Dense)", "稀疏检索 (Sparse)"],
                            label="检索方式", value="混合检索 (Hybrid)",
                        )
                        srch_btn = gr.Button("检索", variant="primary")
                        srch_status = gr.Markdown("")
                        srch_results = gr.Dataframe(
                            headers=["得分", "菜名", "类别", "层级", "内容预览"],
                            label="检索结果", wrap=True,
                        )

                    with gr.Tab("数据概览"):
                        data_stats = gr.Markdown("点击 **刷新** 加载数据统计")
                        gr.Markdown("---")
                        gr.Markdown(f"**提示**：如需更新统计，请先运行 `{index_cmd}` 重建索引后点击刷新。")
                        data_refresh = gr.Button("刷新")

                    with gr.Tab("关于"):
                        gr.Markdown(about_md)

            with gr.Column(scale=3, elem_id="chat-col"):
                gr.Markdown("### 对话助手")
                chatbot = gr.Chatbot(
                    label="对话历史", height=480,
                    avatar_images=(None, ""), autoscroll=True,
                )
                chat_input = gr.Textbox(
                    label="输入问题", placeholder="输入你的问题，按 Enter 发送…", container=False,
                )
                with gr.Row():
                    chat_send = gr.Button("发送", variant="primary", scale=1)
                    chat_clear = gr.Button("清除", scale=1, min_width=80)
                with gr.Row():
                    chat_mode_hint = gr.Markdown("当前模式：**规则**")
                    chat_mode_toggle = gr.Checkbox(label="LLM 模式", value=False)

        # =================================================================
        # Events
        # =================================================================

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
                [c.get("dish", "?"), c.get("level", ""), c.get("section", ""), c.get("category", "")]
                for c in trace.get("chunks", [])
            ]
            answer = trace.get("answer", "（无结果）")
            return intent_data, gr.Dataframe(value=rows, headers=["菜名", "层级", "章节", "类别"]), answer

        pipe_btn.click(
            fn=on_pipe_run,
            inputs=[pipe_query, pipe_top_k, pipe_llm],
            outputs=[pipe_intent, pipe_results, pipe_answer],
            concurrency_limit=1,
            show_progress="full",
        )

        def on_srch(q, k, mode):
            if not q.strip():
                return "请输入查询", gr.Dataframe(value=[])
            status, rows, _ = _search(q, k, mode)
            return status, gr.Dataframe(
                value=rows, headers=["得分", "菜名", "类别", "层级", "内容预览"], wrap=True)

        srch_btn.click(
            fn=on_srch,
            inputs=[srch_query, srch_k, srch_mode],
            outputs=[srch_status, srch_results],
            concurrency_limit=1,
            show_progress="full",
        )

        def on_chat_msg(msg, history, llm, top_k):
            return _chat_answer(msg, history, llm, top_k)

        chat_send.click(
            fn=on_chat_msg,
            inputs=[chat_input, chatbot, llm_state, top_k_state],
            outputs=[chatbot, chat_input],
            concurrency_limit=1,
            show_progress="full",
        )
        chat_input.submit(
            fn=on_chat_msg,
            inputs=[chat_input, chatbot, llm_state, top_k_state],
            outputs=[chatbot, chat_input],
            concurrency_limit=1,
            show_progress="full",
        )
        chat_clear.click(fn=lambda: [], outputs=[chatbot])

        def on_llm_toggle(v):
            return v, "当前模式：**LLM**" if v else "当前模式：**规则**"

        pipe_llm.change(fn=on_llm_toggle, inputs=[pipe_llm], outputs=[llm_state, chat_mode_hint])
        chat_mode_toggle.change(fn=on_llm_toggle, inputs=[chat_mode_toggle], outputs=[llm_state, chat_mode_hint])
        pipe_llm.change(fn=lambda v: v, inputs=[pipe_llm], outputs=[chat_mode_toggle])
        chat_mode_toggle.change(fn=lambda v: v, inputs=[chat_mode_toggle], outputs=[pipe_llm])
        pipe_top_k.change(fn=lambda v: v, inputs=[pipe_top_k], outputs=[top_k_state])

        data_refresh.click(fn=_load_stats, outputs=[data_stats], concurrency_limit=1, show_progress="full")

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recipe RAG System")
    parser.add_argument("--backend", choices=("src", "langchain"), default="src",
                        help="Backend implementation (default: src)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    _backend = args.backend
    _init_logging()
    log = logging.getLogger("app")
    log.info("Starting with backend=%s", _backend)

    if _backend == "langchain":
        from src_langchain.config import APP_HOST, APP_PORT, APP_QUEUE_DEFAULT_CONCURRENCY, APP_QUEUE_MAX_SIZE
    else:
        from src.config import APP_HOST, APP_PORT, APP_QUEUE_DEFAULT_CONCURRENCY, APP_QUEUE_MAX_SIZE

    _warmup()

    demo = _build_ui()
    demo.queue(
        default_concurrency_limit=APP_QUEUE_DEFAULT_CONCURRENCY,
        max_size=APP_QUEUE_MAX_SIZE,
    ).launch(
        share=args.share,
        server_name=APP_HOST, server_port=APP_PORT,
    )
