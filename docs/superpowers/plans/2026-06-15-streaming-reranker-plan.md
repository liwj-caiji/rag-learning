# Streaming + Cross-Encoder Reranker 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 添加 SSE 流式输出 + Cross-Encoder 重排序，提升 RAG 系统检索精度和用户体验

**Architecture:** 流式输出在 pipeline 层新增 `run_stream()` 生成器，底层 LLM 调用 `stream=True` 逐 token yield。Reranker 在 RRF 融合后插入 Cross-Encoder 精排，对 howto/ingredient 意图开启。双后端（src/ 和 src_langchain/）同步改动。

**Tech Stack:** FastAPI · sse-starlette · sentence-transformers (CrossEncoder) · BGE-reranker-v2-minicpm · DeepSeek API streaming

---

## File Map

| 操作 | 文件 | 职责 |
|------|------|------|
| Modify | `src/config.py` | 新增 streaming + reranker 配置 |
| Modify | `src_langchain/config.py` | 同上，LangChain 后端 |
| Create | `src/retrieval/reranker.py` | CrossEncoderReranker 单例 |
| Modify | `src/retrieval/hybrid.py` | `hybrid_search()` 加 `rerank` 参数 |
| Modify | `src/generation/llm_generator.py` | 新增 `generate_stream()` |
| Modify | `src/generation/pipeline.py` | 新增 `run_stream()`，howto/ingredient 开 rerank |
| Create | `src_langchain/retrieval/reranker.py` | LangChain 版 CrossEncoderReranker |
| Modify | `src_langchain/retrieval/ensemble.py` | `hybrid_search()` 加 `rerank` 参数 |
| Modify | `src_langchain/generation/llm_chain.py` | 新增 `generate_stream()` |
| Modify | `src_langchain/pipeline.py` | 新增 `run_stream()`，howto/ingredient 开 rerank |
| Create | `server.py` | FastAPI SSE 入口 |
| Modify | `app.py` | Gradio Chat 流式接入 |
| Modify | `scripts/evaluate.py` | 新增 `--rerank` flag |
| Create | `tests/test_reranker.py` | Reranker 单测 |
| Create | `tests/test_streaming.py` | 流式输出单测 |
| Modify | `docs/rag-resume.md` | 简历更新 |
| Modify | `docs/rag-resume-project-summary.md` | 项目总结更新 |
| Modify | `requirements.txt` | 新增 fastapi, uvicorn, sse-starlette |

---

### Task 1: 新增配置项（双后端）

**Files:**
- Modify: `src/config.py`
- Modify: `src_langchain/config.py`

- [ ] **Step 1: 在 `src/config.py` 末尾添加 streaming 和 reranker 配置**

在 `src/config.py` 末尾追加：

```python
# ============================================================================
# Streaming
# ============================================================================

LLM_GEN_STREAM = True
LLM_GEN_STREAM_TIMEOUT = 30.0

# ============================================================================
# Cross-Encoder Reranker
# ============================================================================

RERANK_ENABLED = True
RERANK_MODEL = "BAAI/bge-reranker-v2-minicpm"
RERANK_CANDIDATES_N = 30    # number of RRF-fused candidates to feed into reranker
RERANK_MAX_LENGTH = 512      # max token length for CrossEncoder input
```

- [ ] **Step 2: 在 `src_langchain/config.py` 末尾添加相同配置**

在 `src_langchain/config.py` 末尾追加相同内容。

- [ ] **Step 3: 验证配置导入**

```bash
cd /c/Users/liwenjie/Desktop/rag && .venv/Scripts/activate && python -c "from src.config import RERANK_MODEL, LLM_GEN_STREAM; print(RERANK_MODEL, LLM_GEN_STREAM)"
```

Expected: `BAAI/bge-reranker-v2-minicpm True`

- [ ] **Step 4: Commit**

```bash
git add src/config.py src_langchain/config.py
git commit -m "feat: add streaming and Cross-Encoder reranker configuration"
```

---

### Task 2: Cross-Encoder Reranker — 原生后端

**Files:**
- Create: `src/retrieval/reranker.py`
- Modify: `src/retrieval/hybrid.py`
- Modify: `src/generation/pipeline.py`

- [ ] **Step 1: 创建 `src/retrieval/reranker.py`**

```python
"""Cross-Encoder reranker for post-RRF fine-grained relevance scoring."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.config import RERANK_MODEL, RERANK_MAX_LENGTH, RERANK_CANDIDATES_N

_reranker_instance: Optional[CrossEncoderReranker] = None


def get_reranker() -> CrossEncoderReranker:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance


class CrossEncoderReranker:
    """Re-score candidates with a Cross-Encoder model for precise relevance.

    Unlike Bi-Encoders (dual-tower: query & chunk encoded separately →
    cosine), a Cross-Encoder concatenates (query, chunk) and runs full
    self-attention across both. This captures fine-grained semantic
    interaction at the cost of higher latency per pair — which is why
    we only apply it to the top-N RRF candidates, not the full index.
    """

    def __init__(
        self,
        model_name: str = RERANK_MODEL,
        max_length: int = RERANK_MAX_LENGTH,
    ):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_name, max_length=max_length)

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        """Re-score candidates and return top_k sorted by relevance.

        Args:
            query: Original user query.
            candidates: List of {score, chunk} dicts (RRF-fused).
            top_k: Number of results to return after reranking.

        Returns:
            Candidates sorted by Cross-Encoder relevance score desc,
            with `score` updated to the CE score and `rrf_score`
            preserved.
        """
        if not candidates:
            return []

        # Build (query, chunk_text) pairs
        pairs = [
            (query, c["chunk"].get("text", ""))
            for c in candidates
        ]

        # Cross-Encoder scoring
        ce_scores = self._model.predict(pairs, show_progress_bar=False)

        # Single score → wrap in list
        if not hasattr(ce_scores, "__len__"):
            ce_scores = [ce_scores]

        # Attach CE scores and re-sort
        for c, ce_score in zip(candidates, ce_scores):
            c["rrf_score"] = c["score"]       # preserve RRF score
            c["ce_score"] = float(ce_score)
            c["score"] = float(ce_score)       # primary score ← CE

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]
```

- [ ] **Step 2: 修改 `src/retrieval/hybrid.py` — `hybrid_search()` 加 rerank 参数**

修改 `hybrid_search()` 函数签名和末尾逻辑：

```python
def hybrid_search(
    query: str,
    k: int = HYBRID_TOPK_DEFAULT,
    dense_k: int = DENSE_CANDIDATES_K,
    sparse_k: int = SPARSE_CANDIDATES_K,
    rrf_k: float = RRF_K,
    rerank: bool = False,
    rerank_top_n: int = 30,
) -> List[Dict]:
```

在 `return fused[:k]` 之前插入 rerank 逻辑。找到 `fused.sort(key=lambda x: x["score"], reverse=True)` 这一行（约第 159 行），将之后的 `return fused[:k]` 替换为：

```python
    # Sort by RRF score descending
    fused.sort(key=lambda x: x["score"], reverse=True)

    # Optional: Cross-Encoder reranking for fine-grained relevance
    if rerank:
        from .reranker import get_reranker
        candidates_for_rerank = fused[:rerank_top_n]
        fused = get_reranker().rerank(query, candidates_for_rerank, top_k=k)
        return fused

    return fused[:k]
```

- [ ] **Step 3: 修改 `src/generation/pipeline.py` — howto/ingredient 开 rerank**

在 `_retrieve()` 方法中，howto 分支的 `hybrid_search()` 调用加 `rerank=True`：

找到 `PIPELINE_HOWTO_K` 那行（约第 182 行），在 `hybrid_search()` 调用的参数末尾加 `rerank=True`：

```python
            results = hybrid_search(
                query_str, k=PIPELINE_HOWTO_K,
                dense_k=PIPELINE_HOWTO_DENSE_K, sparse_k=PIPELINE_HOWTO_SPARSE_K,
                rerank=True,
            )
```

对 ingredient 分支同样处理（约第 212 行）：

```python
            results = hybrid_search(
                query_str, k=PIPELINE_INGREDIENT_K,
                dense_k=PIPELINE_INGREDIENT_DENSE_K, sparse_k=PIPELINE_INGREDIENT_SPARSE_K,
                rerank=True,
            )
```

- [ ] **Step 4: 验证 reranker 导入和基本功能**

```bash
cd /c/Users/liwenjie/Desktop/rag && .venv/Scripts/activate && python -c "from src.retrieval.reranker import get_reranker; r = get_reranker(); print('Reranker loaded:', r._model.model_name)"
```

Expected: 下载模型（首次），打印 `Reranker loaded: BAAI/bge-reranker-v2-minicpm`

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/reranker.py src/retrieval/hybrid.py src/generation/pipeline.py
git commit -m "feat: add Cross-Encoder reranker for howto/ingredient retrieval"
```

---

### Task 3: Cross-Encoder Reranker — LangChain 后端

**Files:**
- Create: `src_langchain/retrieval/reranker.py`
- Modify: `src_langchain/retrieval/ensemble.py`
- Modify: `src_langchain/pipeline.py`

- [ ] **Step 1: 创建 `src_langchain/retrieval/reranker.py`**

```python
"""Cross-Encoder reranker for post-RRF fine-grained relevance scoring (LangChain backend)."""

from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document

from src_langchain.config import RERANK_MODEL, RERANK_MAX_LENGTH, RERANK_CANDIDATES_N

_reranker_instance: Optional["CrossEncoderReranker"] = None


def get_reranker() -> "CrossEncoderReranker":
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance


class CrossEncoderReranker:
    """Re-score LangChain Documents with a Cross-Encoder model."""

    def __init__(
        self,
        model_name: str = RERANK_MODEL,
        max_length: int = RERANK_MAX_LENGTH,
    ):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_name, max_length=max_length)

    def rerank(
        self,
        query: str,
        candidates: List[Document],
        top_k: int,
    ) -> List[Document]:
        if not candidates:
            return []

        pairs = [(query, doc.page_content) for doc in candidates]

        ce_scores = self._model.predict(pairs, show_progress_bar=False)

        if not hasattr(ce_scores, "__len__"):
            ce_scores = [ce_scores]

        for doc, ce_score in zip(candidates, ce_scores):
            doc.metadata["rrf_score"] = doc.metadata.get("rrf_score", 0)
            doc.metadata["ce_score"] = float(ce_score)

        # Sort by CE score descending
        scored = list(zip(ce_scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]
```

- [ ] **Step 2: 修改 `src_langchain/retrieval/ensemble.py` — `hybrid_search()` 加 rerank 参数**

修改 `hybrid_search()` 函数签名和末尾：

```python
def hybrid_search(
    query: str,
    k: int = HYBRID_TOPK_DEFAULT,
    dense_k: int = DENSE_CANDIDATES_K,
    sparse_k: int = SPARSE_CANDIDATES_K,
    rrf_k: float = RRF_K,
    rerank: bool = False,
    rerank_top_n: int = 30,
) -> List[Document]:
```

将 `return retriever.invoke(query)` 替换为：

```python
    retriever = RRFFusionRetriever(
        k=(rerank_top_n if rerank else k),
        dense_k=dense_k, sparse_k=sparse_k, rrf_k=rrf_k,
    )
    results = retriever.invoke(query)

    if rerank:
        from .reranker import get_reranker
        results = get_reranker().rerank(query, results, top_k=k)

    return results
```

- [ ] **Step 3: 修改 `src_langchain/pipeline.py` — howto/ingredient 开 rerank**

在 `_retrieve()` 方法中，howto 分支的 `hybrid_search()` 调用加 `rerank=True`（约第 177 行）：

```python
            results = hybrid_search(
                query_str, k=PIPELINE_HOWTO_K,
                dense_k=PIPELINE_HOWTO_DENSE_K, sparse_k=PIPELINE_HOWTO_SPARSE_K,
                rerank=True,
            )
```

ingredient 分支同样（约第 189 行）：

```python
            results = hybrid_search(
                query_str, k=PIPELINE_INGREDIENT_K,
                dense_k=PIPELINE_INGREDIENT_DENSE_K, sparse_k=PIPELINE_INGREDIENT_SPARSE_K,
                rerank=True,
            )
```

- [ ] **Step 4: 验证 LangChain 后端 reranker 导入**

```bash
cd /c/Users/liwenjie/Desktop/rag && .venv/Scripts/activate && python -c "from src_langchain.retrieval.reranker import get_reranker; r = get_reranker(); print('LC Reranker loaded')"
```

Expected: `LC Reranker loaded`

- [ ] **Step 5: Commit**

```bash
git add src_langchain/retrieval/reranker.py src_langchain/retrieval/ensemble.py src_langchain/pipeline.py
git commit -m "feat: add Cross-Encoder reranker for LangChain backend"
```

---

### Task 4: LLM 流式生成 — 原生后端

**Files:**
- Modify: `src/generation/llm_generator.py`
- Modify: `src/generation/pipeline.py`

- [ ] **Step 1: 在 `src/generation/llm_generator.py` 添加 `generate_stream()` 方法**

在 `LLMGenerator` 类中，`generate()` 方法之后添加：

```python
    def generate_stream(self, query: str, context: List[Dict], intent: str, target_dish: Optional[str] = None):
        """Stream answer tokens one at a time via SSE.

        Yields str tokens. Falls back to non-streaming if LLM is unavailable.
        """
        if not self._client:
            fallback_answer = self._fallback_gen.generate(query, context, intent, target_dish=target_dish) if self._fallback_gen else ""
            yield fallback_answer
            return

        if not context:
            yield self._empty_response(intent, target_dish=target_dish)
            return

        system_prompt = self._build_system_prompt(intent, target_dish=target_dish)
        context_text = self._format_context(context, intent)
        user_prompt = self._build_user_prompt(query, context_text, intent)

        from src.config import LLM_GEN_STREAM_TIMEOUT

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=LLM_GEN_TEMPERATURE,
                max_tokens=LLM_GEN_MAX_TOKENS,
                timeout=LLM_GEN_STREAM_TIMEOUT,
                stream=True,
            )
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception:
            fallback_answer = self._fallback_gen.generate(query, context, intent, target_dish=target_dish) if self._fallback_gen else ""
            yield fallback_answer
```

- [ ] **Step 2: 在 `src/generation/pipeline.py` 添加 `run_stream()` 方法**

在 `RAGPipeline` 类中，`trace()` 方法之后添加：

```python
    def run_stream(self, query: str, top_k: int = 5):
        """Execute pipeline with streaming answer tokens.

        Yields dicts: {"stage": "rewrite|retrieve|generate|done", ...}
        The generate stage yields {"stage": "generate", "token": str}.
        """
        import time
        t0 = time.time()

        # Stage 1: Rewrite
        intent_result = self.rewriter.rewrite(query)
        yield {
            "stage": "rewrite",
            "intent": intent_result.intent,
            "target_dish": intent_result.target_dish,
            "rewritten": intent_result.rewritten,
            "filters": intent_result.filters,
            "probes": intent_result.probes,
        }

        # Stage 2: Retrieve
        t1 = time.time()
        context = self._retrieve(intent_result, top_k, query)
        yield {
            "stage": "retrieve",
            "num_chunks": len(context),
            "chunks": [
                {
                    "dish": c["chunk"]["metadata"].get("dish_name") or "",
                    "level": c["chunk"].get("level") or "",
                    "section": c["chunk"]["metadata"].get("section_type") or "",
                    "category": c["chunk"]["metadata"].get("category") or "",
                }
                for c in context
            ],
            "elapsed": round(time.time() - t1, 3),
        }

        # Stage 3: Generate (streaming)
        full_answer = []
        for token in self.generator.generate_stream(
            query, context, intent_result.intent,
            target_dish=intent_result.target_dish,
        ):
            full_answer.append(token)
            yield {"stage": "generate", "token": token}

        # Stage 4: Done
        yield {
            "stage": "done",
            "answer": "".join(full_answer),
            "total_elapsed": round(time.time() - t0, 3),
        }
```

- [ ] **Step 3: 验证流式生成**

```bash
cd /c/Users/liwenjie/Desktop/rag && .venv/Scripts/activate && python -c "
from src.generation import RAGPipeline
pipe = RAGPipeline(use_llm=True)
for event in pipe.run_stream('麻婆豆腐怎么做', top_k=3):
    if event['stage'] == 'generate':
        print(event['token'], end='', flush=True)
    elif event['stage'] == 'done':
        print(f'\n\nElapsed: {event[\"total_elapsed\"]}s')
"
```

Expected: 逐字输出回答，末尾打印耗时。

- [ ] **Step 4: Commit**

```bash
git add src/generation/llm_generator.py src/generation/pipeline.py
git commit -m "feat: add streaming generation with SSE token yield"
```

---

### Task 5: LLM 流式生成 — LangChain 后端

**Files:**
- Modify: `src_langchain/generation/llm_chain.py`
- Modify: `src_langchain/pipeline.py`

- [ ] **Step 1: 在 `src_langchain/generation/llm_chain.py` 添加 `generate_stream()` 方法**

在 `LLMGenerator` 类（注意：这是 LangChain 后端的，不是 Task 4 那个）的 `_llm_generate()` 方法之后添加：

```python
    def generate_stream(self, query: str, context: List[Document], intent: str, target_dish: Optional[str] = None):
        """Stream answer tokens via ChatOpenAI.stream()."""
        if not self._llm:
            fallback_answer = self._fallback_gen.generate(query, context, intent, target_dish=target_dish) if self._fallback_gen else ""
            yield fallback_answer
            return

        if not context:
            yield self._empty_response(intent, target_dish=target_dish)
            return

        system_prompt = get_system_prompt(intent, target_dish=target_dish)
        context_text = format_context(context)
        user_prompt = build_user_prompt(query, context_text)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{user_prompt}"),
        ])

        from ..config import LLM_GEN_STREAM_TIMEOUT
        streaming_llm = ChatOpenAI(
            model=self.model,
            base_url=self.api_base,
            api_key=self._llm.openai_api_key,
            temperature=LLM_GEN_TEMPERATURE,
            max_tokens=LLM_GEN_MAX_TOKENS,
            timeout=LLM_GEN_STREAM_TIMEOUT,
        )

        chain = prompt | streaming_llm | StrOutputParser()
        invoke_config = {}
        if self.callbacks:
            invoke_config["callbacks"] = self.callbacks

        try:
            for chunk in chain.stream({"user_prompt": user_prompt}, config=invoke_config):
                yield chunk
        except Exception:
            fallback_answer = self._fallback_gen.generate(query, context, intent, target_dish=target_dish) if self._fallback_gen else ""
            yield fallback_answer
```

- [ ] **Step 2: 在 `src_langchain/pipeline.py` 添加 `run_stream()` 方法**

在 `trace()` 方法之后添加：

```python
    @observe(name="RAGPipeline.run_stream")
    def run_stream(self, query: str, top_k: int = 5):
        """Execute pipeline with streaming answer tokens."""
        t0 = time.time()

        intent_result = self.rewriter.classify(query)
        yield {
            "stage": "rewrite",
            "intent": intent_result.intent,
            "target_dish": intent_result.target_dish,
            "rewritten": intent_result.rewritten,
            "filters": intent_result.filters,
            "probes": intent_result.probes,
        }

        t1 = time.time()
        context = self._retrieve(intent_result, top_k, query)
        yield {
            "stage": "retrieve",
            "num_chunks": len(context),
            "chunks": [
                {
                    "dish": doc.metadata.get("dish_name") or "",
                    "level": doc.metadata.get("level") or "",
                    "section": doc.metadata.get("section_type") or "",
                    "category": doc.metadata.get("category") or "",
                }
                for doc in context
            ],
            "elapsed": round(time.time() - t1, 3),
        }

        full_answer = []
        for token in self.generator.generate_stream(
            query, context, intent_result.intent,
            target_dish=intent_result.target_dish,
        ):
            full_answer.append(token)
            yield {"stage": "generate", "token": token}

        answer = "".join(full_answer)
        total_elapsed = time.time() - t0
        self._enrich_trace(query, intent_result, context, answer, total_elapsed)

        yield {
            "stage": "done",
            "answer": answer,
            "total_elapsed": round(total_elapsed, 3),
        }
```

- [ ] **Step 3: 验证 LangChain 流式生成**

```bash
cd /c/Users/liwenjie/Desktop/rag && .venv/Scripts/activate && python -c "
from src_langchain.pipeline import RAGPipeline
pipe = RAGPipeline(use_llm=True)
for event in pipe.run_stream('红烧肉怎么做', top_k=3):
    if event['stage'] == 'generate':
        print(event['token'], end='', flush=True)
    elif event['stage'] == 'done':
        print(f'\n\nElapsed: {event[\"total_elapsed\"]}s')
"
```

Expected: 逐字输出回答。

- [ ] **Step 4: Commit**

```bash
git add src_langchain/generation/llm_chain.py src_langchain/pipeline.py
git commit -m "feat: add streaming generation for LangChain backend"
```

---

### Task 6: FastAPI 服务器

**Files:**
- Create: `server.py`
- Modify: `requirements.txt`

- [ ] **Step 1: 更新 `requirements.txt`**

在 `requirements.txt` 末尾追加：

```
fastapi>=0.115.0
uvicorn>=0.30.0
sse-starlette>=2.0.0
```

安装新依赖：

```bash
cd /c/Users/liwenjie/Desktop/rag && .venv/Scripts/activate && uv pip install fastapi uvicorn sse-starlette && uv pip freeze > requirements.txt 2>/dev/null || pip install fastapi uvicorn sse-starlette
```

- [ ] **Step 2: 创建 `server.py`**

```python
"""
Recipe RAG System — FastAPI Server with SSE streaming.

Usage:
    python server.py                        # default: src backend
    python server.py --backend langchain    # LangChain backend
    python server.py --port 8080
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

# Backend selection
_backend: str = "src"


def _resolve(name: str):
    import importlib
    return importlib.import_module(f"{_backend}.{name}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _init_logging():
    cfg = _resolve("config")
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL),
        format=cfg.LOG_FORMAT,
        datefmt=cfg.LOG_DATE_FORMAT,
    )
    fh = logging.FileHandler(cfg.LOG_FILE, encoding="utf-8", mode="a")
    fh.setFormatter(logging.Formatter(cfg.LOG_FORMAT))
    logging.getLogger().addHandler(fh)
    for noisy in cfg.LOG_SUPPRESS:
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Pipeline cache
# ---------------------------------------------------------------------------

_pipeline_cache = {}


def _get_pipeline(use_llm: bool = True):
    key = f"pipe_{use_llm}"
    if key not in _pipeline_cache:
        log = logging.getLogger("server")
        log.info("Creating pipeline (use_llm=%s, backend=%s)", use_llm, _backend)
        mod = _resolve("pipeline")
        _pipeline_cache[key] = mod.RAGPipeline(use_llm=use_llm)
    return _pipeline_cache[key]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Recipe RAG API",
    description="Streaming RAG API for Chinese recipe Q&A",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "backend": _backend}


@app.post("/chat")
async def chat(query: str = Query(..., description="User query"), top_k: int = Query(5, ge=1, le=20)):
    """Stream RAG pipeline results via SSE."""

    async def event_stream():
        log = logging.getLogger("server")
        log.info("Chat stream start | query=%r top_k=%d", query, top_k)
        try:
            pipe = _get_pipeline(use_llm=True)
            for event in pipe.run_stream(query, top_k=top_k):
                yield {"data": json.dumps(event, ensure_ascii=False)}
        except Exception as e:
            log.error("Chat stream failed: %s", e)
            yield {
                "event": "error",
                "data": json.dumps({"stage": "error", "message": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_stream())


@app.post("/search")
async def search(query: str = Query(...), k: int = Query(5, ge=1, le=20), rerank: bool = Query(False)):
    """Non-streaming hybrid search."""
    log = logging.getLogger("server")
    mod = _resolve("retrieval")
    results = mod.hybrid_search(query, k=k, rerank=rerank)
    return {
        "query": query,
        "num_results": len(results),
        "results": [
            {
                "score": r.get("score", 0),
                "dish_name": r["chunk"]["metadata"].get("dish_name", ""),
                "category": r["chunk"]["metadata"].get("category", ""),
                "level": r["chunk"].get("level", ""),
                "text_preview": r["chunk"].get("text", "")[:200],
            }
            for r in results
        ],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recipe RAG API Server")
    parser.add_argument("--backend", choices=("src", "langchain"), default="src")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    _backend = args.backend
    _init_logging()

    log = logging.getLogger("server")
    log.info("Starting server | backend=%s host=%s port=%d", _backend, args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
```

- [ ] **Step 3: 启动 FastAPI 并验证健康检查**

启动服务器（后台）：

```bash
cd /c/Users/liwenjie/Desktop/rag && .venv/Scripts/activate && python server.py &
sleep 3
curl http://127.0.0.1:8000/health
```

Expected: `{"status":"ok","backend":"src"}`

- [ ] **Step 4: 测试 SSE 流式接口**

```bash
curl -X POST "http://127.0.0.1:8000/chat?query=麻婆豆腐怎么做&top_k=3" -N
```

Expected: 多行 `data:` SSE 事件流，包含 rewrite/retrieve/generate/done 阶段。

- [ ] **Step 5: 停止后台服务器并 Commit**

```bash
kill %1 2>/dev/null || true
git add server.py requirements.txt
git commit -m "feat: add FastAPI server with SSE streaming endpoint"
```

---

### Task 7: Gradio UI 流式接入

**Files:**
- Modify: `app.py`

- [ ] **Step 1: 修改 `app.py` 的 `_chat_answer` 函数为流式生成器**

找到 `_chat_answer` 函数（约第 116 行），替换为：

```python
def _chat_answer(message: str, history: List, use_llm: bool, top_k: int):
    """Streaming chat — yields updated history after each token."""
    if not message or not message.strip():
        yield history, ""
        return
    
    log.info("Chat start (stream) | query=%r use_llm=%s top_k=%d", message, use_llm, top_k)
    t0 = time.time()
    
    if not use_llm:
        # Rule mode: non-streaming
        try:
            pipe = _get_pipeline(use_llm=False)
            answer = pipe.run(message, top_k=top_k)
            elapsed = time.time() - t0
            log.info("Chat done (rule) | elapsed=%.2fs answer_len=%d", elapsed, len(answer))
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": answer})
            yield history, ""
        except Exception as e:
            log.error("Chat failed: %s", e)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"错误：{e}"})
            yield history, ""
        return

    # LLM mode: streaming
    try:
        pipe = _get_pipeline(use_llm=True)
        history.append({"role": "user", "content": message})
        
        full_answer = ""
        for event in pipe.run_stream(message, top_k=top_k):
            if event["stage"] == "generate":
                full_answer += event["token"]
                # Yield updated history with partial answer
                partial_history = list(history)
                partial_history.append({"role": "assistant", "content": full_answer + " ▌"})
                yield partial_history, ""
            elif event["stage"] == "done":
                elapsed = event.get("total_elapsed", time.time() - t0)
                log.info("Chat done (stream) | elapsed=%.2fs answer_len=%d", elapsed, len(full_answer))
                history.append({"role": "assistant", "content": full_answer})
                yield history, ""
                return
            elif event["stage"] == "error":
                history.append({"role": "assistant", "content": f"错误：{event.get('message', '')}"})
                yield history, ""
                return
    except Exception as e:
        elapsed = time.time() - t0
        log.error("Chat stream failed after %.2fs | %s: %s", elapsed, type(e).__name__, e)
        history.append({"role": "assistant", "content": f"错误：{e}"})
        yield history, ""
```

- [ ] **Step 2: 修改 chat_send 和 chat_input.submit 的事件绑定**

找到 `chat_send.click` 和 `chat_input.submit`（约第 444-457 行），确保它们的事件绑定正确。当前的绑定已经 OK，但检查 `_chat_answer` 返回值解包是否正确——现在它是 generator，Gradio 会自动处理。

当前绑定代码保持不变即可，Gradio 的 `click` / `submit` 事件原生支持 generator 函数。

- [ ] **Step 3: 启动 Gradio 验证流式聊天**

```bash
cd /c/Users/liwenjie/Desktop/rag && .venv/Scripts/activate && python app.py &
sleep 5
echo "Gradio should be running at http://127.0.0.1:7860"
```

手动在浏览器测试：打开 http://127.0.0.1:7860 → 对话助手 tab → 开启 LLM 模式 → 输入"麻婆豆腐怎么做" → 观察是否逐字输出。

- [ ] **Step 4: 停止并 Commit**

```bash
kill %1 2>/dev/null || true
git add app.py
git commit -m "feat: add streaming chat to Gradio UI"
```

---

### Task 8: 评估对比脚本

**Files:**
- Modify: `scripts/evaluate.py`

- [ ] **Step 1: 在 `scripts/evaluate.py` 添加 `--rerank` flag**

找到 `add_argument` 区域（约第 62-70 行），在 `--verbose` 之前添加：

```python
    parser.add_argument(
        "--rerank", action="store_true", default=False,
        help="Enable Cross-Encoder reranking in retrieval.",
    )
```

- [ ] **Step 2: 注入 rerank 参数到 pipeline 的检索**

当前评估通过 `pipeline.trace()` 或 `pipeline.run()` 收集结果。需要在创建 pipeline 后，通过 monkey-patching 开启 rerank。

在 `main()` 函数的 `pipeline = RAGPipeline(...)` 之后（约第 105 行），添加：

```python
    # Enable reranking if requested
    if args.rerank:
        _original_retrieve = pipeline._retrieve

        def _retrieve_with_rerank(intent_result, top_k, query=""):
            # Monkey-patch: inject rerank into hybrid_search calls
            import src.retrieval.hybrid as hybrid_mod
            _original_hybrid = hybrid_mod.hybrid_search

            def _hybrid_with_rerank(query_str, k=5, dense_k=20, sparse_k=20, rrf_k=60.0, rerank=False, rerank_top_n=30):
                return _original_hybrid(query_str, k=k, dense_k=dense_k, sparse_k=sparse_k, rrf_k=rrf_k, rerank=True, rerank_top_n=rerank_top_n)

            hybrid_mod.hybrid_search = _hybrid_with_rerank
            try:
                return _original_retrieve(intent_result, top_k, query)
            finally:
                hybrid_mod.hybrid_search = _original_hybrid

        pipeline._retrieve = _retrieve_with_rerank
        log.info("Cross-Encoder reranking ENABLED for evaluation")
```

- [ ] **Step 3: 运行 Before 评估（无 rerank）**

```bash
cd /c/Users/liwenjie/Desktop/rag && .venv/Scripts/activate && python scripts/evaluate.py --mode llm --limit 10 --output eval_before.json
```

记录 RAGAS 指标。

- [ ] **Step 4: 运行 After 评估（有 rerank）**

```bash
cd /c/Users/liwenjie/Desktop/rag && .venv/Scripts/activate && python scripts/evaluate.py --mode llm --limit 10 --rerank --output eval_after.json
```

- [ ] **Step 5: 对比结果**

```bash
cd /c/Users/liwenjie/Desktop/rag && .venv/Scripts/activate && python -c "
import json
for label, path in [('Before', 'eval_before.json'), ('After', 'eval_after.json')]:
    with open(path) as f:
        data = json.load(f)
    metrics = data.get('metrics', data)
    print(f'=== {label} Rerank ===')
    for k, v in metrics.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
"
```

- [ ] **Step 6: Commit**

```bash
git add scripts/evaluate.py
git commit -m "feat: add --rerank flag for before/after evaluation comparison"
```

---

### Task 9: 单元测试

**Files:**
- Create: `tests/test_reranker.py`
- Create: `tests/test_streaming.py`

- [ ] **Step 1: 创建 `tests/test_reranker.py`**

```python
"""Tests for Cross-Encoder reranker."""

import pytest


class TestCrossEncoderReranker:
    """Unit tests for CrossEncoderReranker."""

    @pytest.fixture
    def sample_candidates(self):
        return [
            {
                "score": 0.032,
                "chunk": {
                    "text": "麻婆豆腐是一道经典的川菜，主要原料包括豆腐、牛肉末、豆瓣酱、花椒粉。",
                    "level": "dish",
                    "metadata": {"dish_name": "麻婆豆腐", "category": "meat_dish"},
                },
            },
            {
                "score": 0.028,
                "chunk": {
                    "text": "红烧肉是一道著名的上海菜，以五花肉为主要原料，配以酱油、糖、料酒等调料。",
                    "level": "dish",
                    "metadata": {"dish_name": "红烧肉", "category": "meat_dish"},
                },
            },
            {
                "score": 0.025,
                "chunk": {
                    "text": "清炒时蔬是一道简单的素菜，需要新鲜蔬菜和蒜末。",
                    "level": "dish",
                    "metadata": {"dish_name": "清炒时蔬", "category": "vegetable_dish"},
                },
            },
        ]

    def test_reranker_returns_results(self, sample_candidates):
        """Reranker should return top_k results with updated scores."""
        from src.retrieval.reranker import get_reranker
        reranker = get_reranker()

        results = reranker.rerank("麻婆豆腐怎么做", sample_candidates, top_k=2)
        assert len(results) == 2
        for r in results:
            assert "score" in r
            assert "rrf_score" in r
            assert "ce_score" in r

    def test_reranker_preserves_rrf_score(self, sample_candidates):
        """RRF score should be preserved in rrf_score field."""
        from src.retrieval.reranker import get_reranker
        reranker = get_reranker()

        original_scores = [c["score"] for c in sample_candidates]
        results = reranker.rerank("麻婆豆腐怎么做", sample_candidates, top_k=3)

        for orig, result in zip(original_scores, results):
            assert result["rrf_score"] == orig

    def test_reranker_sorts_by_ce_score(self, sample_candidates):
        """Results should be sorted by CE score descending."""
        from src.retrieval.reranker import get_reranker
        reranker = get_reranker()

        results = reranker.rerank("麻婆豆腐怎么做", sample_candidates, top_k=3)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), f"Scores not descending: {scores}"

    def test_reranker_first_result_most_relevant(self, sample_candidates):
        """Most relevant candidate (麻婆豆腐) should rank first for 麻婆豆腐 query."""
        from src.retrieval.reranker import get_reranker
        reranker = get_reranker()

        results = reranker.rerank("麻婆豆腐怎么做", sample_candidates, top_k=3)
        assert "麻婆豆腐" in results[0]["chunk"]["metadata"]["dish_name"]

    def test_reranker_empty_candidates(self):
        """Empty candidate list should return empty."""
        from src.retrieval.reranker import get_reranker
        reranker = get_reranker()

        results = reranker.rerank("test", [], top_k=5)
        assert results == []

    def test_reranker_top_k_larger_than_candidates(self, sample_candidates):
        """top_k > len(candidates) should return all candidates."""
        from src.retrieval.reranker import get_reranker
        reranker = get_reranker()

        results = reranker.rerank("test", sample_candidates, top_k=10)
        assert len(results) == len(sample_candidates)

    def test_reranker_singleton(self):
        """get_reranker() should return the same instance."""
        from src.retrieval.reranker import get_reranker
        r1 = get_reranker()
        r2 = get_reranker()
        assert r1 is r2
```

- [ ] **Step 2: 运行 reranker 测试**

```bash
cd /c/Users/liwenjie/Desktop/rag && .venv/Scripts/activate && python -m pytest tests/test_reranker.py -v
```

Expected: 7 tests pass.

- [ ] **Step 3: 创建 `tests/test_streaming.py`**

```python
"""Tests for streaming generation."""

import pytest


class TestStreamingGeneration:
    """Unit tests for streaming LLM generation."""

    @pytest.fixture
    def sample_context(self):
        return [
            {
                "chunk": {
                    "text": "麻婆豆腐做法：1.豆腐切块 2.炒牛肉末 3.加豆瓣酱 4.加花椒粉 5.出锅。",
                    "level": "section",
                    "metadata": {
                        "dish_name": "麻婆豆腐",
                        "category": "meat_dish",
                        "section_type": "操作",
                    },
                },
            },
        ]

    def test_generate_stream_returns_tokens(self, sample_context):
        """generate_stream should yield string tokens."""
        import os
        if not os.environ.get("DEEPSEEK_API_KEY"):
            pytest.skip("DEEPSEEK_API_KEY not set")

        from src.generation.llm_generator import LLMGenerator
        generator = LLMGenerator()

        tokens = list(generator.generate_stream(
            "麻婆豆腐怎么做", sample_context, "howto", target_dish="麻婆豆腐",
        ))
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
        # Combined answer should be non-empty
        answer = "".join(tokens)
        assert len(answer) > 0

    def test_generate_stream_empty_context(self):
        """Empty context should return empty/fallback response."""
        import os
        if not os.environ.get("DEEPSEEK_API_KEY"):
            pytest.skip("DEEPSEEK_API_KEY not set")

        from src.generation.llm_generator import LLMGenerator
        generator = LLMGenerator()

        tokens = list(generator.generate_stream(
            "麻婆豆腐怎么做", [], "howto", target_dish="麻婆豆腐",
        ))
        assert len(tokens) == 1  # single fallback message

    def test_generate_stream_no_api_key(self):
        """Without API key, should fallback to template."""
        import os
        # Temporarily hide API key
        saved_key = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            from src.generation.llm_generator import LLMGenerator
            generator = LLMGenerator(api_key=None)

            tokens = list(generator.generate_stream(
                "麻婆豆腐怎么做", [
                    {"chunk": {"text": "...", "level": "dish",
                     "metadata": {"dish_name": "麻婆豆腐"}}},
                ], "howto",
            ))
            assert len(tokens) >= 1
        finally:
            if saved_key:
                os.environ["DEEPSEEK_API_KEY"] = saved_key


class TestPipelineStream:
    """Integration tests for RAGPipeline.run_stream()."""

    def test_run_stream_returns_stages(self):
        """run_stream should yield rewrite, retrieve, generate, done stages."""
        import os
        if not os.environ.get("DEEPSEEK_API_KEY"):
            pytest.skip("DEEPSEEK_API_KEY not set")

        from src.generation import RAGPipeline
        pipe = RAGPipeline(use_llm=True)

        stages_seen = set()
        for event in pipe.run_stream("麻婆豆腐怎么做", top_k=2):
            stages_seen.add(event["stage"])

        assert "rewrite" in stages_seen
        assert "retrieve" in stages_seen
        assert "generate" in stages_seen
        assert "done" in stages_seen

    def test_run_stream_rewrite_has_intent(self):
        """Rewrite stage should include intent info."""
        import os
        if not os.environ.get("DEEPSEEK_API_KEY"):
            pytest.skip("DEEPSEEK_API_KEY not set")

        from src.generation import RAGPipeline
        pipe = RAGPipeline(use_llm=True)

        for event in pipe.run_stream("麻婆豆腐怎么做", top_k=2):
            if event["stage"] == "rewrite":
                assert "intent" in event
                assert event["intent"] in ("howto", "recommendation", "ingredient", "factual")
                break

    def test_run_stream_retrieve_has_chunks(self):
        """Retrieve stage should include chunk metadata."""
        import os
        if not os.environ.get("DEEPSEEK_API_KEY"):
            pytest.skip("DEEPSEEK_API_KEY not set")

        from src.generation import RAGPipeline
        pipe = RAGPipeline(use_llm=True)

        for event in pipe.run_stream("麻婆豆腐怎么做", top_k=2):
            if event["stage"] == "retrieve":
                assert "num_chunks" in event
                assert "chunks" in event
                assert isinstance(event["chunks"], list)
                break

    def test_run_stream_done_has_answer(self):
        """Done stage should include full answer and elapsed time."""
        import os
        if not os.environ.get("DEEPSEEK_API_KEY"):
            pytest.skip("DEEPSEEK_API_KEY not set")

        from src.generation import RAGPipeline
        pipe = RAGPipeline(use_llm=True)

        done_event = None
        for event in pipe.run_stream("今天吃什么", top_k=2):
            if event["stage"] == "done":
                done_event = event
                break

        assert done_event is not None
        assert "answer" in done_event
        assert "total_elapsed" in done_event
        assert len(done_event["answer"]) > 0
```

- [ ] **Step 4: 运行 streaming 测试**

```bash
cd /c/Users/liwenjie/Desktop/rag && .venv/Scripts/activate && python -m pytest tests/test_streaming.py -v -k "not no_api_key"
```

Expected: 需要 `DEEPSEEK_API_KEY` 的测试通过，`no_api_key` 测试通过。

- [ ] **Step 5: 运行全量测试确保无回归**

```bash
cd /c/Users/liwenjie/Desktop/rag && .venv/Scripts/activate && python -m pytest tests/ -v --ignore=tests/test_reranker.py --ignore=tests/test_streaming.py -k "not index_available"
```

Expected: 现有单元测试全部通过。

- [ ] **Step 6: Commit**

```bash
git add tests/test_reranker.py tests/test_streaming.py
git commit -m "test: add unit tests for reranker and streaming generation"
```

---

### Task 10: 简历更新

**Files:**
- Modify: `docs/rag-resume.md`
- Modify: `docs/rag-resume-project-summary.md`

- [ ] **Step 1: 更新 `docs/rag-resume.md`**

在现有内容基础上：

a) 更新技术栈标签行（第 3 行），加 `FastAPI`、`BGE-Reranker`、`SSE`：

```markdown
**Python · FAISS · BM25 · RRF · BGE-Reranker · LangChain · DeepSeek · FastAPI · SSE · Gradio · RAGAS · Langfuse**
```

b) 在"分块与 Small-to-Big"一节（第 1 节）之后，插入新节：

```markdown
## 2. 多阶段检索 — RRF + Cross-Encoder 精排

三阶段检索架构：FAISS 稠密检索（Bi-Encoder）+ BM25 稀疏检索 → RRF 融合（k=60）→ Cross-Encoder 精排（BGE-reranker-v2-minicpm）。Bi-Encoder 双塔模型独立编码 query 和 chunk 后求余弦相似度，速度快但交互不充分；Cross-Encoder 将 (query, chunk) 拼接送入 Transformer 全注意力计算，精度高但速度慢。因此采用粗排→精排的两阶段策略：RRF 融合 30-40 个候选后交给 Cross-Encoder 精排取 top-k，兼顾效率与精度。

针对不同意图差异化路由：howto/ingredient 开精排（需要精确匹配菜名和操作），recommendation 保持 MMR 多样性重排（需要品类多样性而非单点精度）。
```

c) 原"混合检索 — RRF 融合"节（第 2 节）改为第 3 节，更新内容加 reranker 说明。

d) 在 LangChain 节之后插入流式输出节：

```markdown
## 5. SSE 流式输出与 FastAPI 服务化

DeepSeek API `stream=True` 逐 token 返回，Pipeline 层新增 `run_stream()` 生成器方法，yield 四阶段事件（rewrite / retrieve / generate / done）。FastAPI + sse-starlette 封装 `POST /chat` SSE 接口，Gradio Chat 组件接入流式，用户即刻看到逐字输出。对比常规同步生成 5-15s 的空白等待，流式显著改善交互体验。同时保留非流式 `run()` 方法作为规则模式兜底。
```

e) 面试 QA 新增：

```markdown
### Q9: 为什么 RRF 之后还要加 Cross-Encoder Reranker？

**答**：RRF 融合的是 Bi-Encoder (FAISS) 和 BM25 的排序结果，两者都不是精细的相关性判断——Bi-Encoder 独立编码 query 和 chunk，只有余弦相似度一个标量交互；BM25 是词袋匹配。Cross-Encoder 将 `(query, chunk)` 拼接后做全自注意力，每个 token 都能与另一句的每个 token 交互，相关度判断远更精确。但 Cross-Encoder 慢（每对都需完整前向），不能在全量索引上用，所以只在 RRF top-30 候选上精排。这和多阶段推荐系统"召回→粗排→精排"的思路一致。

### Q10: SSE vs WebSocket，为什么选用 SSE？

**答**：这个场景是单向推送（服务端→客户端推送 token），不需要客户端向服务端持续发送数据。SSE 基于 HTTP 协议，实现简单（标准 EventSource API），天然支持自动重连，不需要额外的心跳保活机制。WebSocket 适合双向实时通信（如聊天室），对于 LLM 流式输出是杀鸡用牛刀，还会增加运维复杂度（代理配置、连接管理）。
```

- [ ] **Step 2: 更新 `docs/rag-resume-project-summary.md`**

a) 技术栈表新增行：

在现有技术栈表中添加：
```
| 重排序 | Cross-Encoder (bge-reranker-v2-minicpm) | RRF 融合后精排，提升 howto/ingredient 精度 |
| 流式输出 | SSE + FastAPI + sse-starlette | 服务化部署，逐 token 流式返回 |
```

b) 架构图追加 Reranker 环节（在第 3 节架构核心设计）：

在 RRF 融合之后插入一行：
```
│   Cross-Encoder Reranker                     │
│   BGE-reranker-v2-minicpm 精排 (howto/ingr)   │
```

c) 关键设计决策表新增：

```
| **RRF + Cross-Encoder 两阶段检索** | RRF 粗排 + CE 精排，平衡效率与精度，类似推荐系统召回→粗排→精排架构 |
| **SSE 流式输出** | 单向推送场景，比 WebSocket 轻量，自动重连，实现简单 |
```

d) 项目规模表更新代码量：

```
| 代码量 | ~4000 行（双后端 + 共享模块 + FastAPI server） |
```

e) 扩展方向更新（第 8 题），标记流式输出为已完成：

```
| 流式输出（SSE/WebSocket） | FastAPI + SSE 流式输出已实现 |
```

- [ ] **Step 3: Commit**

```bash
git add docs/rag-resume.md docs/rag-resume-project-summary.md
git commit -m "docs: update resume with streaming and Cross-Encoder reranker"
```

---

## Summary

| Task | 内容 | 依赖 |
|------|------|------|
| 1 | 配置项（双后端） | 无 |
| 2 | Reranker 原生后端 | Task 1 |
| 3 | Reranker LangChain 后端 | Task 1 |
| 4 | Streaming 原生后端 | Task 1 |
| 5 | Streaming LangChain 后端 | Task 1 |
| 6 | FastAPI server | Task 4 |
| 7 | Gradio UI 流式 | Task 4 |
| 8 | 评估对比脚本 | Task 2 |
| 9 | 单元测试 | Tasks 2-5 |
| 10 | 简历更新 | Tasks 2-7 |
