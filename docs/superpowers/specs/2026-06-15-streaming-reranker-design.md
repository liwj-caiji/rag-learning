# Streaming + Cross-Encoder Reranker 设计

## 目标

1. SSE 流式输出 — 消除长回答等待体验
2. Cross-Encoder 重排序 — 提升 howto/ingredient 意图的检索精度
3. 简历更新 — 对齐大模型应用工程师岗位

## 1. 流式输出 (SSE)

### 现状

`LLMGenerator._llm_generate()` 使用 `chat.completions.create()` 同步返回完整回答，用户等待 5-15s。

### 改动

**`src/generation/llm_generator.py`** — 新增 `generate_stream()`

- 复用现有 `_build_system_prompt()` / `_build_user_prompt()`
- `chat.completions.create(stream=True)` → yield token
- 保留 `generate()` 作为非流式兜底

**`src/generation/pipeline.py`** — 新增 `run_stream()`

- 生成器函数，yield 中间状态 + token
- 状态结构：`{"stage": "rewrite|retrieve|generate|done", ...}`

**`app.py`** — Gradio Chat 接入流式

- `_chat_answer` 改为 generator
- Chatbot 组件逐 token 更新

**`server.py`** — 新增 FastAPI 入口

- `POST /chat` SSE streaming
- `GET /health` 健康检查

**`src_langchain/`** — 同步适配

- `ChatOpenAI.stream()` 天然支持

### 配置新增

```python
# src/config.py
LLM_GEN_STREAM = True
LLM_GEN_STREAM_TIMEOUT = 30.0  # 流式超时更长
```

## 2. Cross-Encoder 重排序

### 现状

FAISS dense + BM25 sparse → RRF fusion → top-k。RRF 仅依赖排名，无精细语义相关度计算。

### 方案：RRF 后加 Cross-Encoder 精排

```
FAISS dense (20) ─┐
                   ├─ RRF fusion (30~40) ─→ Cross-Encoder → top-k
BM25 sparse (20) ─┘
```

Cross-Encoder 将 (query, chunk_text) 拼接送入 Transformer 全注意力计算，比 Bi-Encoder 的余弦相似度精确。

### 改动

**`src/retrieval/reranker.py`** — 新增

```python
class CrossEncoderReranker:
    model_name = "BAAI/bge-reranker-v2-minicpm"
    max_length = 512
    
    def rerank(self, query, candidates, top_k) -> List[Dict]
```

**`src/config.py`** — 配置新增

```python
RERANK_ENABLED = True
RERANK_MODEL = "BAAI/bge-reranker-v2-minicpm"
RERANK_CANDIDATES_N = 30  # RRF 后送入 reranker 的候选数
RERANK_MAX_LENGTH = 512
```

**`src/retrieval/hybrid.py`** — 修改 `hybrid_search()`

- 新增 `rerank=False` 参数
- RRF 融合后，取 top-N 送 reranker，返回精排后的 top-k

**`src/generation/pipeline.py`** — 修改 `_retrieve()`

- howto / ingredient 开启 rerank
- recommendation 不开启（已有 MMR 多样性重排）
- factual 可选

**`scripts/evaluate.py`** — 新增 `--rerank` flag

- Before/After 评估对比
- 输出 RAGAS 5 指标差异

**`src_langchain/`** — 同步适配

- `ContextualCompressionRetriever` 或自建 compressor

### 模型选型

`BAAI/bge-reranker-v2-minicpm`：中文好，体积约 1GB，延迟低，适合 CPU 运行。

## 3. 简历更新

- 新增"流式输出"和"Cross-Encoder 重排序"两个技术点
- 面试 QA 新增：
  - SSE vs WebSocket 选型理由
  - Cross-Encoder vs Bi-Encoder 区别
  - 为什么 RRF 之后还要 Reranker
- 更新技术栈表，加 `FastAPI`、`BGE-Reranker`、`sse-starlette`

## 依赖新增

```
# requirements.txt
fastapi>=0.115.0
uvicorn>=0.30.0
sse-starlette>=2.0.0
sentence-transformers  # 已有，CrossEncoder 包含在内
```

## 不改动的

- 知识库索引结构不变
- 评估数据集不变
- 规则模式不涉及流式
- 不做多轮对话
- 不做 Docker 化

## 工作量估计

| 模块 | 预估 |
|------|------|
| LLMGenerator.generate_stream() | 1h |
| RAGPipeline.run_stream() | 1h |
| Gradio 流式接入 | 1h |
| FastAPI server.py | 1h |
| CrossEncoderReranker | 1.5h |
| hybrid_search 接入 | 0.5h |
| pipeline 接入 | 0.5h |
| 评估 Before/After | 1.5h |
| LangChain 同步 | 1h |
| 简历更新 | 1h |
| **合计** | **~10h** |
