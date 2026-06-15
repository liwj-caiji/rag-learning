# 食谱 RAG 系统 — 项目总结（简历用）

> 适用岗位：AI 应用开发 / 大模型应用工程师 / RAG 系统开发

## 1. 一句话概述

基于 **混合检索（FAISS + BM25 + RRF）+ LLM 生成** 的中文食谱 RAG 问答系统，363+ 道菜谱知识库，支持规则/LLM 双模式，双后端架构（原生 SDK / LangChain），Gradio Web UI，集成 RAGAS 评估和 Langfuse 可观测性。

---

## 2. 技术栈

| 层次 | 技术 | 用途 |
|------|------|------|
| 分块 | 层级 Markdown 分块（L1/L2/L3） | 按 `#` `##` `###` 标题拆分为 dish/section/subsection |
| Embedding | SentenceTransformers (text2vec-base-chinese, 768d) | 中文语义向量编码 |
| 稠密检索 | FAISS (IndexFlatIP + L2 normalize) | 余弦相似度检索 |
| 稀疏检索 | BM25Okapi + jieba 分词 | 中文关键词匹配 |
| 融合 | RRF (Reciprocal Rank Fusion, k=60) | 稠密+稀疏结果融合 |
| 意图分类 | 规则匹配 / DeepSeek JSON mode | 4 意图分类 + 约束提取 + 查询改写 |
| 答案生成 | 模板引擎 / DeepSeek API | 结构化/自然语言回答 |
| 评估 | RAGAS (5 指标) | context_precision/recall, faithfulness, relevancy, correctness |
| 可观测 | Langfuse (@observe + CallbackHandler) | Pipeline trace + LLM 调用追踪 |
| 重排序 | Cross-Encoder (bge-reranker-v2-minicpm-layerwise) | RRF 融合后精排，提升 howto/ingredient 精度 |
| 流式输出 | SSE + FastAPI + sse-starlette | 服务化部署，逐 token 流式返回 |
| UI | Gradio | 交互式问答 + 检索演示 + 数据概览 |
| 编排 | LangChain (可选后端) | BaseRetriever, ChatOpenAI, StrOutputParser |

---

## 3. 架构核心设计

```
用户查询
    │
    ▼
┌─────────────────────────────────────────────┐
│  QueryRewriter                              │
│  意图分类 + 约束提取 + 多探针查询改写          │
├─────────────────────────────────────────────┤
│  HybridRetriever                            │
│  FAISS (dense) + BM25 (sparse) → RRF 融合    │
│  意图路由：推荐→多探针, 做法→偏重操作章节       │
├─────────────────────────────────────────────┤
│   Cross-Encoder Reranker                     │
│   BGE-reranker-v2-minicpm 精排 (howto/ingr)   │
├─────────────────────────────────────────────┤
│  Context Enrichment                         │
│  howto/ingredient: 注入完整食谱源文件          │
├─────────────────────────────────────────────┤
│  Generator                                  │
│  模板渲染 / LLM (DeepSeek)                   │
└─────────────────────────────────────────────┘
    │
    ▼
  Gradio Web UI  /  CLI 评估
```

### 关键设计决策

| 决策 | 理由 |
|------|------|
| **双后端架构**（原生 SDK + LangChain） | 原生后端深入理解 RAG 底层；LangChain 后端对接标准化生态和 Langfuse |
| **规则/LLM 双模式** | 规则模式零依赖离线可用；LLM 模式提供更智能的理解和生成 |
| **RRF 融合而非线性加权** | 稠密/稀疏分数尺度不同，RRF 仅依赖排名，无需分数归一化 |
| **意图驱动检索路由** | 不同意图需要不同检索策略和参数（推荐多探针、做法偏重操作章节） |
| **完整食谱上下文注入** | LLM 回答做法需要原料+步骤完整信息，分块片段易缺失 |
| **层级 Markdown 分块** | 食谱 `#` `##` `###` 结构天然适合层级分块，粒度精确控制 |
| **MMR + 类别轮询多样性** | 推荐场景需品类多样性，同品类结果对用户价值低 |
| **共享模块提取** | jieba 分词和 RAGAS 评估原本在 4 处重复定义，提取至 `shared/` |
| **集中式配置** | 所有可调参数集中于 `src/config.py`，避免分散定义导致不一致 |
| **RRF + Cross-Encoder 两阶段检索** | RRF 粗排 + CE 精排，平衡效率与精度，类似推荐系统召回→粗排→精排架构 |
| **SSE 流式输出** | 单向推送场景，比 WebSocket 轻量，自动重连，实现简单 |

---

## 4. 核心实现细节

### 4.1 层级 Markdown 分块

```python
class RecipeSplitter:
    """每个 .md 文件产出最多 3 层 chunk"""
    def split(self) -> List[Dict]:
        # L1-dish:    # 菜名 + 导言（介绍、难度、卡路里）
        # L2-section: ## 章节（必备原料和工具、操作、附加内容）
        # L3-subsection: ### 子步骤（仅 ## 操作 内部拆分）
```

从导言正则提取结构化元数据：
- 烹饪难度：`★` ~ `★★★★`
- 卡路里：数值 → low (≤300) / high (≥600)

### 4.2 混合检索（FAISS + BM25 + RRF）

```python
def hybrid_search(query, k=5, dense_k=20, sparse_k=20, rrf_k=60.0):
    dense_results = dense_search(query, k=dense_k)    # FAISS 余弦相似度
    sparse_results = sparse_search(query, k=sparse_k)  # BM25 + jieba
    # RRF 融合: score = Σ 1/(K + rank_i) for each channel
    # 按 RRF 分数降序 → top-k
```

- Embedding 模型：`shibing624/text2vec-base-chinese`，768 维
- BM25 分词：jieba.lcut()，过滤空 token
- RRF 常数 K=60（经典值）

### 4.3 意图驱动多探针推荐

```python
def recommend_dishes(query, k=5, filters=None, diversify=True, probes=None):
    # 1. Multi-probe: 多个搜索探针 × hybrid_search(k=15) → 去重
    # 2. 元数据过滤 (difficulty, category, calories)
    # 3. 多样性重排:
    #    - 优先 MMR (query embedding + 元数据相似度)
    #    - 回退 类别轮询 (round-robin by category)
```

- 搜索探针：从原始查询生成 3-5 个不同角度的搜索词
- MMR λ=0.5，平衡相关性和多样性
- 相似度启发式（无 embedding 回退）：同菜 0.9，同品类 0.3，不同 0.0

### 4.4 完整食谱上下文注入

针对 howto/ingredient 场景，仅靠分块片段（如 `## 操作` 不含原料）导致 LLM 回答不完整：

```python
def _enrich_with_full_recipe(results, dish_query):
    # 从 chunk metadata.path → 读取源 .md 文件完整内容
    # 作为 {level: "full_recipe", section_type: "完整食谱"} 条目
    # 插入 chunk 列表头部，LLM 优先使用
```

### 4.5 双后端架构

```
app.py --backend src|langchain
    │
    ├── src/ (原生)
    │   ├── FAISS + SentenceTransformer 手动编码
    │   ├── BM25Okapi + jieba
    │   ├── OpenAI SDK → DeepSeek API
    │   └── 模板渲染 / LLM 生成
    │
    └── src_langchain/ (LangChain)
        ├── FAISS.from_documents() + HuggingFaceEmbeddings
        ├── BM25Retriever + jieba
        ├── ChatOpenAI → DeepSeek API
        └── ChatPromptTemplate + StrOutputParser
```

| 维度 | src/ | src_langchain/ |
|------|------|---------------|
| 检索器 | 自定义函数 | BaseRetriever / RRFFusionRetriever |
| LLM 调用 | 原生 OpenAI SDK | ChatOpenAI + StrOutputParser |
| 意图分类 | 规则 + JSON mode 手动解析 | ChatOpenAI.with_structured_output() |
| 可观测性 | 结构化日志 | Langfuse @observe + CallbackHandler |
| 文件数 | ~15 个 | ~15 个（镜像结构） |

### 4.6 RAGAS 评估流水线

```python
class RAGASEvaluator:
    def evaluate(self, samples, metrics) -> EvaluationResult:
        for sample in samples:
            trace = self.pipeline.trace(sample.query)  # 收集回答+上下文
        ragas_result = ragas.evaluate(dataset, metrics,
            llm=self.eval_llm, embeddings=self.eval_embeddings)
        # → context_precision / recall / faithfulness / relevancy / correctness
```

- 评估数据集：30 条中文测试查询（YAML），覆盖 4 种意图，含 ground truth
- 中文适配：`adapt_prompts(language="chinese")` + pickle 缓存
- 评估器与后端解耦：pipeline 通过构造器注入

### 4.7 Langfuse 可观测性（LangChain 后端）

```python
# tracing.py — 全局单例 CallbackHandler
langfuse_handler = LangchainCallbackHandler()

# pipeline.py — @observe 自动创建 trace
@observe(name="RAGPipeline.run")
def run(self, query): ...

# llm_chain.py — callbacks 传递链
llm.invoke(prompt, config={"callbacks": [langfuse_handler]})
```

- Pipeline 级 trace：捕获 query → answer 全链路
- LLM 级 span：自动捕获每次 LLM 调用的 token 用量、模型名、延迟
- Trace 富化：注入 intent、target_dish、rewritten_query、contexts 等元数据

---

## 5. 难点与解决方案

| 难点 | 原因 | 解决方案 |
|------|------|---------|
| **LLM 不知道原料信息** | howto 检索到的 `## 操作` 分块不含原料 | `_enrich_with_full_recipe()` 注入完整食谱源文件 |
| **LLM 编造不存在的菜品** | 检索未命中但 LLM 凭参数知识"回答" | System prompt 强约束"只使用上下文信息"，未命中转"不知道+推荐" |
| **稠密/稀疏分数不可比** | FAISS 余弦相似度和 BM25 分数量纲完全不同 | RRF 仅依赖排名，无需分数归一化 |
| **推荐结果同质化** | 检索返回同品类多个菜品 | MMR 重排 + 类别轮询确保品类多样性 |
| **意图分类边界模糊** | "推荐"和"做法"可能同时出现 | 最长关键词匹配优先 + LLM 结构化输出兜底 |
| **jieba 分词函数 4 处重复** | 双后端发展过程中各自复制 | 提取至 `shared/tokenizer.py` 统一 |
| **评估模块与后端耦合** | 原始评估器硬依赖 `src/` 的 pipeline | 改为构造器注入 pipeline，同一套评估代码评估两种后端 |
| **Gradio UI 启动卡死** | `demo.load` 触发扫描 363 个文件耗时 8s | 改为懒加载 + 内存缓存 |
| **RAGAS 英文 prompt 评估中文效果差** | 评估 LLM 用英文 prompt 评判中文回答 | `adapt_prompts(language="chinese")` + pickle 缓存 |
| **LangChain API 兼容性** | LangChain 0.3+ 移除了多个旧 API | `get_relevant_documents()` → `invoke()`, `create_documents()` → `split_text()` |

---

## 6. 面试可能考察点

### Q1: 为什么用 RRF 融合而不是线性加权？

**答**：稠密检索（FAISS 余弦相似度）和稀疏检索（BM25 分数）的量纲完全不同——FAISS 分数在 [-1, 1]，BM25 分数无上界且受文档长度影响。直接加权需要做分数归一化，而归一化参数的选取高度依赖数据集。RRF（Reciprocal Rank Fusion）仅依赖各通道中的排名，天然消除了量纲差异，且 K=60 是经过广泛验证的经典值。

### Q2: 为什么做双后端而不是只保留一种？

**答**：两个后端解决不同需求。原生后端（`src/`）手动实现 FAISS/BM25/RRF/LLM 调用，用于深入理解 RAG 底层机制；LangChain 后端（`src_langchain/`）使用标准化组件，用于学习 LangChain 生态和对接 Langfuse 可观测性。`app.py --backend` 和共享模块（`shared/`）保证了维护成本可控。从评估结果看，两者的 `context_precision`、`context_recall`、`answer_relevancy` 完全一致。

### Q3: 意图分类为什么设计规则和 LLM 两条路径？

**答**：规则路径零外部依赖、零延迟、可离线运行——适合高频推荐查询（如"今天吃什么"）。LLM 路径能理解复杂查询（如"我想做一道不太辣的、适合夏天吃的川菜"）中隐含的多个约束。实际部署中规则先行筛选，LLM 兜底复杂场景。

### Q4: 为什么要在 howto 场景注入完整食谱文件？

**答**：分块后 `## 操作` 片段只包含烹饪步骤，不含原料信息。如果用户问"麻婆豆腐怎么做"，LLM 只看到操作步骤但看不到原料列表，回答就不完整。更糟的是 LLM 可能凭参数知识"编造"原料（如加入知识库菜谱中不存在但"常见"的调料）。解决方案是检测到目标菜名后，直接读取源 `.md` 文件完整内容，作为「完整食谱」条目插入 chunk 列表头部。

### Q5: MMR 重排中相似度怎么算的？为什么不用 embedding？

**答**：主路径使用 query embedding 计算语义相似度。但如果 embedding 模型加载失败（首次运行未缓存），回退到元数据启发式：同菜名→0.9，同品类→0.3，不同→0.0。这基于一个合理假设：用户不想推荐列表里出现同一道菜或全是同一品类的菜。

### Q6: 评估系统怎么保证公正性？

**答**：① 评估器通过构造器注入 pipeline，不绑定具体后端实现；② 30 条测试查询覆盖 4 种意图，包含 boundary case；③ ground truth 与知识库实际食谱文件内容对齐（不是随便写的）；④ RAGAS 指标由独立的评估 LLM（deepseek-chat）计算，不是目标 pipeline 的 LLM 自评；⑤ 中文 prompt 通过 ragas 官方的 `adapt_prompts` 翻译而非手写，避免 prompt 偏差。

### Q7: RAG 回答中最大的可靠性问题是什么？怎么解决的？

**答**：最大问题是 LLM **幻觉**——当检索未命中时，LLM 可能凭参数知识"编造"回答。我们的对抗措施：① System prompt 强约束"只使用提供的上下文信息"；② 在 howto/ingredient 场景检测目标菜名是否在上下文中出现，未出现时引导 LLM 输出"不知道+推荐相似菜品"，而非强行回答；③ RAGAS faithfulness 指标持续监控回答是否忠实于检索上下文。

### Q8: 这个系统的扩展方向？

**答**：① 支持更多 LLM provider（Anthropic, Ollama），当前仅 DeepSeek；② 流式输出—已实现 (FastAPI + SSE)，改善长回答等待体验；③ Query 扩展（HyDE / Query2Doc），提升稀疏检索的召回；④ 搜索结果重排序（Cross-Encoder Reranker），替代 RRF 提升精度；⑤ 知识库增量更新（CDC 监控食谱仓库变更）；⑥ Docker 化部署 + API 化（FastAPI 替换 Gradio）。

---

## 7. 项目规模

| 指标 | 数值 |
|------|------|
| 知识库规模 | 363+ 道中文家常菜 |
| 分块总数 | ~2000+ |
| 向量维度 | 768 |
| 意图类型 | 4 种（推荐/做法/原料/知识） |
| 评估样本 | 30 条（含 ground truth） |
| 评估指标 | 5 项 RAGAS 指标 |
| 代码量 | ~4000 行（双后端 + 共享模块 + FastAPI server） |
| 后端实现 | 2 套（原生 SDK / LangChain） |
| 开发周期 | 约 5 天 |
