# 食谱 RAG 系统 — 架构与实现文档

> 基于混合检索 + LLM 生成的智能食谱问答系统，双后端实现（原生 SDK / LangChain），Gradio Web UI。

## 1. 项目概述

用户输入烹饪相关查询（推荐、做法、原料、知识），系统经过**查询改写 → 混合检索 → 上下文增强 → 答案生成**流水线返回结构化回答。

### 核心设计原则

- **双后端架构**：`src/` 使用原生 SentenceTransformer + FAISS + OpenAI SDK；`src_langchain/` 使用 LangChain 组件封装，通过 `app.py --backend` 切换
- **混合检索 + 精排**：稠密检索（FAISS 余弦相似度）+ 稀疏检索（BM25 jieba 分词）+ RRF 融合 → Cross-Encoder 重排序
- **意图驱动路由**：4 种意图（recommendation / howto / ingredient / factual）各自不同的检索策略、精排和生成模板
- **规则 / LLM 双模式**：意图分类和答案生成均支持规则驱动（零依赖）和 LLM 驱动（DeepSeek API）
- **流式生成**：LLM 模式逐 token 输出，Gradio 前端实时渲染打字机效果
- **共享模块提取**：`shared/` 包中 jieba 分词器和 RAGAS 评估模块被双后端共享

---

## 2. 系统架构

### 2.1 拓扑结构

```
用户输入查询
     │
     ▼
┌──────────────────────────────────────────────────────┐
│                  RAG Pipeline                        │
│                                                      │
│  QueryRewriter ──→ HybridRetriever ──→ Generator     │
│  (意图分类+改写)    (FAISS + BM25 + RRF)  (模板/LLM)  │
│       │                  │                   │        │
│       ▼                  ▼                   ▼        │
│  IntentResult     List[Chunk]          最终回答        │
│  (intent, filters,    │                               │
│   probes, target)     │                               │
│                       ▼                               │
│              _enrich_with_full_recipe()               │
│              (howto/ingredient 场景注入完整食谱源文件)   │
└──────────────────────────────────────────────────────┘
     │
     ▼
  Gradio Web UI 展示
```

### 2.2 Pipeline 四阶段

| 阶段 | 组件 | 输入 | 输出 |
|------|------|------|------|
| **改写** | QueryRewriter / LLMIntentClassifier | 用户原始查询 | IntentResult（意图、改写查询、过滤条件、搜索探针） |
| **检索** | hybrid_search() / recommend_dishes() | IntentResult | 检索到的 chunk 列表 |
| **增强** | _enrich_with_full_recipe() | chunk 列表 + 目标菜名 | 注入完整食谱源文件的增强上下文 |
| **生成** | TemplateGenerator / LLMGenerator | query + context + intent | 自然语言回答 |

---

## 3. 技术栈分层

```
┌──────────────────────────────────────────────────┐
│  app.py              Gradio Web UI               │  展示层
│                      双后端切换 (--backend)        │
├──────────────────────────────────────────────────┤
│  src/generation/     生成层                       │
│  pipeline.py          RAGPipeline (编排)          │
│  llm_generator.py     LLMGenerator (DeepSeek)    │
│  template.py          TemplateGenerator (规则)    │
├──────────────────────────────────────────────────┤
│  src/rewriting/      改写层                       │
│  intent.py            规则意图分类 + 约束提取      │
│  llm_intent.py        LLM 意图分类 (JSON mode)    │
│  rewriter.py          QueryRewriter 抽象          │
├──────────────────────────────────────────────────┤
│  src/retrieval/      检索层                       │
│  hybrid.py            混合检索 + 推荐              │
│  diversity.py         MMR 重排 + 类别轮询         │
│  filters.py           元数据过滤                  │
├──────────────────────────────────────────────────┤
│  src/preprocess/     预处理层                     │
│  splitter.py          层级 Markdown 分块          │
│  indexer.py           FAISS + BM25 索引构建       │
│  config.py            预处理配置                  │
├──────────────────────────────────────────────────┤
│  src/config.py        全局配置 (9 段)             │
├──────────────────────────────────────────────────┤
│  shared/              共享模块                    │
│  tokenizer.py          jieba 中文分词             │
│  evaluation/           RAGAS 评估流水线           │
└──────────────────────────────────────────────────┘
```

---

## 4. 预处理层

### 4.1 数据源

[HowToCook](https://github.com/Anduin2017/HowToCook) 开源食谱库，363+ 道中文家常菜，Markdown 格式，按品类分目录（`aquatic/`、`meat_dish/`、`vegetable_dish/`、`soup/` 等）。

### 4.2 层级分块（RecipeSplitter）

`src/preprocess/splitter.py` — 对每个 `.md` 文件产出最多 3 层 chunk：

| 层级 | 粒度 | 内容 | 元数据 |
|------|------|------|--------|
| **L1-dish** | `# 菜名` heading + 导言 | 菜品介绍、难度、卡路里 | dish_name, category, difficulty, calories |
| **L2-section** | 每个 `##` 章节 | 必备原料和工具、操作、附加内容 | dish_name, section_type |
| **L3-subsection** | `###` 子步骤（仅 `## 操作`） | 单个烹饪步骤 | dish_name, section_type, subsection_name |

分块策略：
- 跳过 `template/` 目录
- 移除标准页脚（PR 引导文字）
- `## 操作` 章节进一步按 `###` 标题拆分为独立步骤 chunk
- 从导言文本正则提取烹饪难度（★~★★★★）和卡路里数值

### 4.3 索引构建（build_index）

`src/preprocess/indexer.py` — 一键构建双通道索引：

```
RecipeSplitter.split() → all_chunks (363+ 文件 → ~2000+ chunks)
    │
    ├── SentenceTransformer(text2vec-base-chinese) → embeddings
    │       └── faiss.normalize_L2() → FAISS IndexFlatIP → faiss_langchain/
    │
    └── jieba.lcut(tokenize) → BM25Okapi → bm25_langchain.pkl
```

- **Embedding 模型**：`shibing624/text2vec-base-chinese`，768 维，L2 归一化后用内积等价余弦相似度
- **BM25**：使用 `rank_bm25` 库，中文文本经 jieba 分词后构建

### 4.4 双后端索引隔离

| 文件 | src/ | src_langchain/ |
|------|------|---------------|
| FAISS 索引 | `faiss.index` | `faiss_langchain/` |
| Chunks | `chunks.pkl` | `chunks_langchain.pkl` |
| BM25 | `bm25_index.pkl` | `bm25_langchain.pkl` |

---

## 5. 检索层

### 5.1 稠密检索（dense_search）

`src/retrieval/hybrid.py:dense_search()` — FAISS 余弦相似度检索：

1. SentenceTransformer 编码查询为 768 维向量
2. L2 归一化 → FAISS `IndexFlatIP` 内积搜索
3. 返回 top-k 候选（默认 20 个）

### 5.2 稀疏检索（sparse_search）

`src/retrieval/hybrid.py:sparse_search()` — BM25 关键词检索：

1. jieba 分词查询
2. `BM25Okapi.get_scores()` 计算每个文档的 BM25 分数
3. 按分数降序返回 top-k 候选（默认 20 个），过滤 `BM25_SCORE_MIN` 以下结果

### 5.3 混合检索（hybrid_search）

RRF（Reciprocal Rank Fusion）融合稠密和稀疏两路结果：

```
dense_results (k=20)      sparse_results (k=20)
       │                          │
       ▼                          ▼
  dense_ranks: {chunk_id: rank}  sparse_ranks: {chunk_id: rank}
       │                          │
       └──────────┬───────────────┘
                  ▼
         RRF(chunk) = Σ 1/(K + rank_i)    K=60
                  │
                  ▼
         按 RRF 分数降序 → 返回 top-k
```

设计要点：
- RRF 常数 K=60（经典值）
- rank 为 1-based（第 1 名 rank=1）

### 5.4 多菜品推荐（recommend_dishes）

`src/retrieval/hybrid.py:recommend_dishes()` — 推荐意图专用检索：

```
1. Multi-probe 候选收集
   多个搜索探针 × hybrid_search(k=15) → 去重 → 仅保留 L1-dish 级 chunk

2. 元数据过滤（difficulty / category / calories / level）

3. 多样性排序
   ├── MMR rerank（基于元数据相似度启发式）
   └── 类别轮询（round-robin by category）
```

### 5.5 多样性控制（diversity.py）

| 方法 | 说明 |
|------|------|
| **mmr_rerank()** | 最大边际相关性重排：`λ * relevance - (1-λ) * max_similarity_to_selected`。λ=0.5 |
| **diversify_by_category()** | 按品类分组后轮询选取，确保推荐结果品类多样 |

相似度启发式（无 embedding 时回退）：
- 同菜名 → 0.9
- 同品类 → 0.3
- 不同品类 → 0.0

### 5.6 元数据过滤（filters.py）

支持的过滤维度：

| 过滤器 | 类型 | 示例 |
|--------|------|------|
| `difficulty` | 前缀匹配 | "★" 匹配 "★★" |
| `category` | 精确匹配 | "meat_dish" |
| `calories` | 数值比较 | `<=300` → low, `>=600` → high |
| `level` | 精确匹配 | "dish" / "section" |
| `target_dish` | 子串匹配 | "麻婆" 匹配 "麻婆豆腐" |

---

## 6. 改写层

### 6.1 意图分类

4 种意图类型：

| 意图 | 说明 | 触发关键词示例 |
|------|------|---------------|
| **recommendation** | 菜品推荐 | "今天吃什么"、"推荐"、"下饭菜" |
| **howto** | 烹饪步骤 | "怎么做"、"步骤"、"如何烧" |
| **ingredient** | 原料清单 | "需要什么材料"、"原料"、"食材" |
| **factual** | 菜品知识 | "是什么"、"哪里菜系"、"区别" |

### 6.2 规则分类器（intent.py）

`classify_intent(query)` — 零依赖意图分类：
- **意图匹配**：关键词最长匹配优先
- **约束提取**：正则提取 difficulty / category / calories
- **菜名提取**：去除意图后缀（"怎么做"、"的做法"）提取目标菜名
- **搜索探针生成**：为推荐意图生成 3-5 个多角度搜索词

### 6.3 LLM 分类器（llm_intent.py）

`LLMIntentClassifier` — 基于 DeepSeek API 的意图分类：

- 使用 `response_format={"type": "json_object"}` 确保结构化输出
- 输出 JSON：`{intent, rewritten_query, filters, target_dish, probes, confidence}`
- 异常/无 API key 时自动回退到规则分类器

```python
# LLM 分类 prompt 核心结构
_SYSTEM_PROMPT = """
分析用户查询 → 输出 JSON {
  intent: recommendation|howto|ingredient|factual,
  rewritten_query: 改写后的搜索查询,
  filters: {difficulty, category, calories},
  target_dish: 目标菜名,
  probes: [搜索探针],
  confidence: 0.0-1.0
}
"""
```

### 6.4 查询改写器（rewriter.py）

```python
class RuleQueryRewriter(BaseQueryRewriter):
    def rewrite(self, query: str) -> IntentResult:
        return classify_intent(query)  # 规则驱动

class LLMQueryRewriter(BaseQueryRewriter):
    def rewrite(self, query: str) -> IntentResult:
        return LLMIntentClassifier().classify(query)  # LLM 驱动
```

---

## 7. 生成层

### 7.1 抽象接口

```python
class Generator(ABC):
    @abstractmethod
    def generate(self, query, context, intent, target_dish) -> str: ...
```

### 7.2 模板生成器（TemplateGenerator）

`src/generation/template.py` — 零 LLM 依赖的结构化回答：

- **recommendation**：列表展示菜品名、类别、难度、卡路里
- **howto**：按菜名分组展示步骤文本；未命中时提示"不知道"并推荐相似菜品
- **ingredient**：展示原料清单；未命中时同样推荐相似菜品
- **factual**：文本摘要展示

### 7.3 LLM 生成器（LLMGenerator）

`src/generation/llm_generator.py` — 基于 DeepSeek API 生成自然语言回答：

- **System Prompt 构建**：根据意图类型注入不同的回答要求
  - howto/ingredient 意图：检测上下文中是否包含目标菜名，区分"直接回答"和"不知道+推荐"两种路径
  - 优先使用「完整食谱」条目（`_enrich_with_full_recipe` 注入）
- **上下文格式化**：`[序号] 菜名 - 章节类型 (子章节)\n内容`
- **流式输出**：`generate_stream()` 通过 `chain.stream()` 逐 token yield，Gradio 前端实时渲染打字机效果
- **异常回退**：API 调用失败时降级为 TemplateGenerator（含流式场景回退）

### 7.4 完整食谱上下文注入（_enrich_with_full_recipe）

`src/generation/pipeline.py:_enrich_with_full_recipe()` — howto/ingredient 场景的特殊优化：

```
retrieved_chunks (可能只是 操作 章节片段)
    │
    ▼
从 chunk metadata.path 提取源文件路径
    │
    ▼
读取 base/HowToCook/dishes/xxx/菜名.md 完整内容
    │
    ▼
作为 {level: "full_recipe", section_type: "完整食谱"} 条目
插入到 chunk 列表头部
    │
    ▼
LLM Generator 收到完整食谱（介绍+原料+步骤）
```

**设计原因**：分块后的 `## 操作` 片段可能不包含原料信息；LLM 回答"怎么做"时需要看到完整食谱，避免编造或遗漏。

---

### 5.7 Cross-Encoder 重排序（reranker.py）

`src_langchain/retrieval/reranker.py` — RRF 融合后对 top-N 候选进行精排：

```
RRF 融合结果 (top-k=30)
       │
       ▼
CrossEncoderReranker.rerank(query, candidates, top_k)
       │
       ├── 构建 (query, chunk_text) 配对
       ├── BAAI/bge-reranker-v2-m3 逐对打分
       └── 按 CE 分数降序 → 返回 top_k
```

设计要点：
- **为何后置**：Cross-Encoder 对每对 (query, chunk) 做全自注意力，计算成本远高于 Bi-Encoder，只在 RRF 后的 top-N 候选上运行
- **模型选择**：`bge-reranker-v2-m3`（568M BERT-based），CPU 上稳定运行，中文效果优秀
- **分数保留**：`rrf_score`（RRF 原始分）和 `ce_score`（Cross-Encoder 分）均写入 metadata，便于分析
- **单例模式**：`get_reranker()` 全局复用，避免重复加载 1.1GB 模型
- **双后端实现**：`src/retrieval/reranker.py`（原生 Dict 格式）+ `src_langchain/retrieval/reranker.py`（LangChain Document 格式）
- **意图路由**：howto / ingredient 意图自动启用（`rerank=True`），如需在 factual 意图启用可扩展

---
## 8. RAG Pipeline

### 8.1 意图驱动检索路由

`src/generation/pipeline.py:_retrieve()` — 根据意图选择不同的检索策略：

| 意图 | 检索策略 | 参数 |
|------|---------|------|
| **recommendation** | `recommend_dishes()` 多探针 + 过滤 + 多样性 | k=5, probes=3-5 |
| **howto** | `hybrid_search()` 偏重「操作」章节 | k=20, dense_k=50, sparse_k=50 |
| **ingredient** | `hybrid_search()` 偏重「必备原料和工具」章节 | k=20, dense_k=50, sparse_k=50 |
| **factual** | `hybrid_search()` 通用检索 | k=5 |

### 8.2 Howto/Ingredient 精排策略

```python
# 优先返回匹配目标菜名 + 目标章节类型的 chunk
section_results = [doc for doc in results
    if doc.section_type == "操作" and dish_query in doc.dish_name]

# 降级：匹配目标菜名的任意 chunk
dish_results = [doc for doc in results if dish_query in doc.dish_name]

# 兜底：取原始检索结果 top-k
```

### 8.3 trace() 调试模式

`pipeline.trace(query)` 返回完整中间结果：

```python
{
    "query": str, "intent": str, "rewritten": str,
    "probes": list, "filters": dict, "target_dish": str,
    "num_chunks": int,
    "chunks": [{"dish", "level", "section", "category", "text"}],
    "answer": str,
}
```

---

## 9. 双后端架构

### 9.1 设计动机

| 维度 | src/（原生） | src_langchain/（LangChain） |
|------|-------------|---------------------------|
| 学习价值 | 理解 RAG 底层机制 | 学习 LangChain 生态 |
| 可维护性 | 较少的抽象层 | 标准化组件，社区认可 |
| 可观测性 | 自定义日志 | Langfuse Callback 自动 tracing |
| 检索器 | 自定义 ToolCollection | LangChain BaseRetriever |
| LLM | 原生 OpenAI SDK | ChatOpenAI + StrOutputParser |

### 9.2 共享模块

`shared/` 包消除代码重复：

| 模块 | 内容 | 使用方 |
|------|------|--------|
| `shared/tokenizer.py` | `chinese_tokenize()` jieba 分词 | 双后端 BM25 |
| `shared/evaluation/` | RAGAS 评估流水线 | 双后端 CLI |

### 9.3 统一入口

`app.py --backend src|langchain` 启动不同后端：

```bash
python app.py                          # 默认 src 后端
python app.py --backend langchain      # LangChain 后端
python app.py --backend langchain --share  # 公网分享
```

内部通过 `_resolve(module_name)` 动态导入对应后端的模块。

---

## 10. 评估系统

### 10.1 RAGAS 评估流水线

`shared/evaluation/` — 与后端解耦的评估模块：

```
test_queries.yaml (30条中文测试查询)
       │
       ▼
  RAGASEvaluator(pipeline)  ← pipeline 通过构造器注入
       │
       ├── pipeline.trace(sample.query) → 收集回答 + 上下文
       │
       ├── RAGAS evaluate() → 计算 5 项指标
       │
       └── print_console_report() / save_json_report()
```

### 10.2 评估指标

| 指标 | 说明 |
|------|------|
| **context_precision** | 检索上下文与问题的精确匹配度 |
| **context_recall** | 检索上下文覆盖 ground truth 的程度 |
| **faithfulness** | 回答是否忠实于检索到的上下文 |
| **answer_relevancy** | 回答与问题的相关性 |
| **answer_correctness** | 回答与 ground truth 的一致性 |

### 10.3 中文提示词适配

使用 ragas `adapt_prompts(language="chinese")` API 自动翻译所有指标提示词为中文，首次运行后 pickle 缓存至 `.ragas_cache/`。

### 10.4 评估数据集

`data/evaluation/test_queries.yaml` — 30 条中文烹饪领域测试查询，覆盖 4 种意图，含 ground truth（与知识库实际食谱内容对齐）。

---

## 11. Langfuse 可观测性（仅 LangChain 后端）

`src_langchain/tracing.py` — Langfuse 集成：

- `@observe` 装饰器自动创建 trace，捕获 pipeline 的 input/output
- `LangchainCallbackHandler` 捕获每次 LLM 调用的 token 用量、模型名、延迟
- Callback 通过 `callbacks` 参数从 Pipeline → IntentClassifier → LLMGenerator 完整传递
- `_enrich_trace()` 向 span 注入元数据（意图、菜名、改写查询、chunk 数、耗时）

---

## 12. CLI 入口 & Web UI

### 12.1 索引构建

```bash
# 原生后端
uv run python -m src.preprocess.indexer

# LangChain 后端
uv run python -m src_langchain.preprocess.indexer
```

### 12.2 评估

```bash
uv run python scripts/evaluate.py --backend src
uv run python scripts/evaluate.py --backend langchain
```

### 12.3 Web UI

```bash
uv run python app.py --backend src
uv run python app.py --backend langchain --share  # 公网分享
```

Gradio 界面包含 4 个 Tab：
- **Pipeline 总览**：完整 RAG 流水线 + 意图分析 + 检索结果 + 生成回答
- **检索演示**：单独测试稠密/稀疏/混合检索
- **数据概览**：食谱统计、品类分布、分块层级分布
- **关于**：技术架构说明

---

## 13. 配置系统

`src/config.py` — 全局唯一配置源，11 个配置段：

| 配置段 | 关键项 |
|--------|--------|
| Paths | `DISHES_DIR`, `VECTORSTORE_DIR`, `FAISS_INDEX_PATH`, `BM25_INDEX_PATH` |
| Embedding | `EMBED_MODEL=shibing624/text2vec-base-chinese`, `EMBED_DIM=768` |
| Chunking | `SKIP_DIRS`, `REMOVE_FOOTER`, `SPLIT_H3` |
| Retrieval | `DENSE_CANDIDATES_K=20`, `SPARSE_CANDIDATES_K=20`, `RRF_K=60` |
| Diversity | `MMR_LAMBDA=0.5`, `SIM_SAME_DISH=0.9` |
| LLM Intent | `LLM_INTENT_MODEL=deepseek-v4-flash`, `LLM_INTENT_TIMEOUT=8s` |
| LLM Gen | `LLM_GEN_MODEL=deepseek-v4-flash`, `LLM_GEN_TIMEOUT=15s` |
| Streaming | `LLM_GEN_STREAM=True`, `LLM_GEN_STREAM_TIMEOUT=30s` |
| Reranker | `RERANK_ENABLED=True`, `RERANK_MODEL=BAAI/bge-reranker-v2-m3`, `RERANK_CANDIDATES_K=30` |
| App | `APP_HOST=127.0.0.1`, `APP_PORT=7860` |
| Logging | `LOG_LEVEL=INFO`, `LOG_FORMAT` |

---

## 14. 项目结构

```
rag/
├── app.py                      # Gradio Web UI 统一入口
├── pyproject.toml              # 项目元数据与依赖声明
├── uv.lock                     # uv 依赖锁文件
├── src/                        # 原生后端
│   ├── config.py               # 全局配置
│   ├── __init__.py
│   ├── preprocess/
│   │   ├── splitter.py         # RecipeSplitter（层级 Markdown 分块）
│   │   ├── indexer.py          # FAISS + BM25 索引构建
│   │   └── config.py           # 向后兼容 re-export
│   ├── retrieval/
│   │   ├── hybrid.py           # dense_search, sparse_search, hybrid_search, recommend_dishes
│   │   ├── diversity.py        # MMR 重排, 类别轮询
│   │   ├── filters.py          # 元数据过滤
│   │   └── reranker.py         # Cross-Encoder 重排序
│   ├── rewriting/
│   │   ├── intent.py           # 规则意图分类 + 约束提取
│   │   ├── llm_intent.py       # LLM 意图分类 (DeepSeek JSON mode)
│   │   └── rewriter.py         # QueryRewriter 抽象
│   └── generation/
│       ├── base.py             # Generator 抽象接口
│       ├── template.py         # TemplateGenerator
│       ├── llm_generator.py    # LLMGenerator (DeepSeek)
│       └── pipeline.py         # RAGPipeline (编排)
├── src_langchain/              # LangChain 后端（镜像结构）
│   ├── config.py
│   ├── preprocess/             # LangChain MarkdownHeaderTextSplitter + FAISS.from_documents
│   ├── retrieval/              # LangChain BaseRetriever + RRF retriever
│   ├── rewriting/              # ChatOpenAI.with_structured_output()
│   ├── generation/             # ChatOpenAI + ChatPromptTemplate + StrOutputParser
│   ├── pipeline.py             # RAGPipeline + Langfuse @observe
│   └── tracing.py              # Langfuse CallbackHandler 初始化
├── shared/                     # 共享模块
│   ├── tokenizer.py            # jieba 中文分词（双后端共用）
│   └── evaluation/             # RAGAS 评估流水线
│       ├── config.py
│       ├── dataset.py          # YAML → EvalSample
│       ├── evaluator.py        # RAGASEvaluator
│       └── reporter.py         # 控制台 + JSON 报告
├── base/HowToCook/             # 食谱数据源（363+ 道中餐）
├── data/
│   ├── vectorstore/            # FAISS + BM25 + chunks 文件
│   └── evaluation/
│       └── test_queries.yaml   # 30 条测试查询
├── scripts/
│   └── evaluate.py             # 评估 CLI
├── docs/                       # 文档
└── logs/                       # 运行日志
```

---

## 15. 已知限制

- 食谱数据源为静态 Markdown 文件集，不支持增量更新
- 混合检索中的相似度启发式（MMR 回退路径）是近似值，非精确语义相似度
- LLM 生成器无 token 计数和成本统计
- ~~无流式输出（streaming），回答需等待完整生成~~（v2.1 已实现）
- 仅支持 DeepSeek 作为 LLM provider（OpenAI 兼容 API），未支持 Anthropic / Ollama
- 无 Docker 化部署方案
- LangChain 后端与原生后端的功能并非 100% 等价（如 LangChain 后端的 RRF 使用 `RRFFusionRetriever`）

---

## 附录 A: 变更记录

### [v2.1] — 2026-06-15

**Added**
- 流式输出（streaming）：`LLMGenerator.generate_stream()` → `chain.stream()` 逐 token yield，Gradio 聊天打字机效果
- Cross-Encoder Reranker（`BAAI/bge-reranker-v2-m3`）：RRF 融合后精排，提升检索精度
- Reranker 单元测试（6 个）+ 流式输出测试（5 个）
- `pyproject.toml`：迁移至 uv 标准项目管理，支持 `uv run`

**Changed**
- 依赖管理：`requirements.txt` → `pyproject.toml`（`uv sync` / `uv run`）
- Reranker 模型：`bge-reranker-v2-minicpm-layerwise`（2.4B LLM，CPU OOM）→ `bge-reranker-v2-m3`（568M BERT，CPU 稳定）
- 固定 `transformers<5.0.0` 以避免兼容性问题

**Fixed**
- Reranker 在 CPU-only 16GB 环境下的稳定性问题

### [v2.0] — 2026-05-29

**Added**
- LangChain 后端 `src_langchain/`：基于 LangChain 组件的完整镜像实现
- Langfuse 可观测性集成（tracing、trace 富化、Callback 传递链）
- `app.py` 双后端统一入口（`--backend src|langchain`）

### [v1.2] — 2026-05-28

**Changed**
- 项目结构优化：`shared/` 提取共享模块（tokenizer + evaluation）
- `src-langchain/rag_langchain/` → `src_langchain/`（消除命名问题和嵌套层级）
- `app.py` 合并两份实现，双后端文件隔离

### [v1.1] — 2026-05-27

**Added**
- RAGAS 评估模块：30 条测试查询 + 5 项指标
- 中文提示词适配（adapt_prompts）
- Ground truth 与食谱文件对齐

### [v1.0] — 2026-05-25~26

**Added**
- 全局配置集中管理（`src/config.py`）
- 完整食谱上下文注入（`_enrich_with_full_recipe`）
- 层级 Markdown 分块（L1/L2/L3）
- 意图驱动检索路由 + 多探针推荐
- MMR 多样性重排 + 类别轮询
- LLM 意图分类（JSON mode）
- 混合检索（FAISS + BM25 + RRF）
- Gradio Web UI

---

## 附录 B: 设计决策

| 决策 | 原因 |
|------|------|
| 双后端架构 | 原生后端用于理解 RAG 底层机制，LangChain 后端用于学习标准化生态 |
| 规则/LLM 双模式 | 规则模式零依赖可离线运行，LLM 模式提供更智能的分类和生成 |
| 层级 Markdown 分块 | 食谱文档的 `#` `##` `###` 结构天然适合层级分块，分块粒度精确 |
| RRF 融合而非线性加权 | 稠密和稀疏分数尺度不同，RRF 仅依赖排名，无需分数归一化 |
| 完整食谱上下文注入 | LLM 回答做法时需要原料+步骤完整信息，仅靠分块片段易缺失和编造 |
| 元数据启发式 MMR 回退 | 每次请求都计算 embedding 开销大，元数据相似度作为轻量替代 |
| 独立配置文件 | 所有可调参数集中于 `src/config.py`，避免分散定义导致不一致 |
| 评估模块与后端解耦 | pipeline 通过构造器注入，同一套评估代码可评估两种后端 |
| jieba 分词函数统一 | 此前在 4 处重复定义，提取至 `shared/tokenizer.py` 消除不一致风险 |
| `--backend` 参数切换而非独立启动文件 | 减少维护负担，避免两份 Gradio 代码不同步 |
