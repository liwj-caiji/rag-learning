# LangChain 重构实现说明

## 概述

将食谱 RAG 系统从自定义实现重构为基于 LangChain 组件的新版本，代码位于 `src_langchain/` 目录。评估模块与分词器提取至 `shared/` 共享包。

## 项目结构

```
rag/
├── app.py                                  # 统一 Web UI 入口 (--backend src|langchain)
├── shared/                                 # 共享包（双后端共用）
│   ├── __init__.py
│   ├── tokenizer.py                        # jieba 中文分词 (chinese_tokenize)
│   └── evaluation/                         # RAGAS 评估模块
│       ├── __init__.py
│       ├── config.py                       # 评估配置 (LLM/指标/批处理)
│       ├── dataset.py                      # YAML 数据集加载 (EvalSample)
│       ├── evaluator.py                    # RAGASEvaluator (pipeline 注入)
│       └── reporter.py                     # 控制台/JSON 报告
├── src/                                    # v1.x 自定义实现
│   ├── config.py
│   ├── generation/
│   ├── preprocess/
│   ├── retrieval/
│   └── rewriting/
└── src_langchain/                          # v2.0 LangChain 实现
    ├── __init__.py
    ├── config.py
    ├── pipeline.py                         # RAG Pipeline 编排
    ├── preprocess/
    │   ├── loader.py                       # DirectoryLoader + TextLoader
    │   ├── splitter.py                     # MarkdownHeaderTextSplitter + 自定义元数据提取
    │   └── indexer.py                      # FAISS + BM25 索引构建
    ├── retrieval/
    │   ├── sparse_retriever.py             # BM25Retriever 封装
    │   ├── ensemble.py                     # RRF 融合自定义 Retriever
    │   ├── diversity.py                    # 类别轮询 + MMR 重排序
    │   └── filters.py                      # 元数据过滤
    ├── rewriting/
    │   ├── intent_types.py                 # IntentResult + 规则匹配逻辑
    │   ├── rule_classifier.py              # RuleIntentClassifier
    │   ├── llm_classifier.py               # LLMIntentClassifier (structured output)
    │   └── rewriter.py                     # 工厂函数
    └── generation/
        ├── base.py                         # Generator 抽象基类
        ├── prompts.py                      # ChatPromptTemplate + format_context()
        ├── template_chain.py               # TemplateGenerator
        └── llm_chain.py                    # LLMGenerator (StrOutputParser)
```

## 模块迁移映射

### 1. 配置层 (`config.py`)

保持与 v1.x 相同的配置结构。LangChain 后端的索引文件使用独立文件名 (`chunks_langchain.pkl`, `bm25_langchain.pkl`, `faiss_langchain/`) 避免与 `src/` 后端冲突。

### 2. 文档加载 (`preprocess/loader.py`)

| v1.x | v2.0 |
|------|------|
| `collect_all_recipes()` — 手动 os.walk + 正则过滤 | `DirectoryLoader(glob="**/*.md")` + `TextLoader` |
| 手动构建路径列表 | `loader.load()` 返回 `List[Document]` |

**使用方式**：
```python
from src_langchain.preprocess import load_recipe_documents
docs = load_recipe_documents()
```

### 3. 文档分块 (`preprocess/splitter.py`)

| v1.x | v2.0 |
|------|------|
| 自定义 `RecipeSplitter` 解析 H1/H2/H3 边界 | `MarkdownHeaderTextSplitter` + `split_text()` |
| 手动行遍历 + `_group_by_headings()` | LangChain 自动按 headers 切分 |
| 手动正则提取 difficulty/calories | `RecipeDocumentTransformer` 在 split 后附加元数据 |

**保留的自定义逻辑**：
- Footer 移除（`FOOTER_RE`）
- H3 子分块（仅在 `## 操作` 下切 `###`）
- difficulty/calories 元数据提取
- dish_name 提取（去除 "的做法" 后缀）

### 4. 索引构建 (`preprocess/indexer.py`)

| v1.x | v2.0 |
|------|------|
| `SentenceTransformer(EMBED_MODEL)` 手动调用 | `HuggingFaceEmbeddings(model_name=EMBED_MODEL)` |
| `faiss.IndexFlatIP` 手动创建、add、序列化 | `FAISS.from_documents(chunks, embeddings, distance_strategy="COSINE")` |
| `BM25Okapi` 手动构建 + pickle | `BM25Retriever.from_documents(chunks, preprocess_func=chinese_tokenize)` |

### 5. 检索层

#### 5.1 BM25 稀疏检索 (`retrieval/sparse_retriever.py`)

```python
from langchain_community.retrievers import BM25Retriever
from shared.tokenizer import chinese_tokenize

bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=chinese_tokenize)
```

#### 5.2 稠密检索 (`retrieval/ensemble.py` 中的 `_load_faiss()`)

```python
from langchain_community.vectorstores import FAISS
faiss_store = FAISS.load_local(FAISS_INDEX_DIR, embeddings, ...)
retriever = faiss_store.as_retriever(search_kwargs={"k": k})
docs = retriever.invoke(query)  # 新版 LangChain API
```

#### 5.3 RRF 融合检索 (`retrieval/ensemble.py` — `RRFFusionRetriever`)

自定义 `RRFFusionRetriever(BaseRetriever)` 覆写 `_get_relevant_documents()`：
- 并行调用 dense (FAISS) 和 sparse (BM25) 通道
- 为每个文档计算 RRF 分数：`1/(k + rank)`
- 合并去重后按 RRF 分数排序返回 Top-K

### 6. 意图分类 (`rewriting/`)

| v1.x | v2.0 |
|------|------|
| `classify_intent()` 关键词匹配 | 不变（`rule_classifier.py`） |
| `OpenAI` 客户端 + `json.loads()` 解析 | `ChatOpenAI` + `with_structured_output(IntentClassificationSchema)` |

```python
from src_langchain.rewriting import RuleIntentClassifier, LLMIntentClassifier

c = RuleIntentClassifier()
result = c.classify("今天吃什么")  # IntentResult(intent="recommendation", ...)

c = LLMIntentClassifier()  # 需要 DEEPSEEK_API_KEY
result = c.classify("今天吃什么")
```

### 7. 答案生成 (`generation/`)

| v1.x | v2.0 |
|------|------|
| f-string 模板 | `ChatPromptTemplate.from_messages(...)` |
| `OpenAI` 客户端手动调用 | `ChatOpenAI + StrOutputParser()` |

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{user_prompt}"),
])
chain = prompt | ChatOpenAI(...) | StrOutputParser()
```

### 8. Pipeline (`pipeline.py`)

与 v1.x 保持相同的编排逻辑：`rewrite → retrieve → enrich → generate`。

```python
from src_langchain.pipeline import RAGPipeline

pipeline = RAGPipeline()           # 规则模式
pipeline = RAGPipeline(use_llm=True)  # LLM 模式
answer = pipeline.run("今天吃什么")
trace = pipeline.trace("麻婆豆腐怎么做", top_k=3)  # 用于评估
```

### 9. 评估 (`shared/evaluation/`)

评估模块提取为 `shared/evaluation/`，通过 pipeline 注入消除对 `src/` 或 `src_langchain/` 的硬依赖：

```python
from shared.evaluation import load_eval_dataset, RAGASEvaluator, print_console_report

pipeline = RAGPipeline(use_llm=True)
samples = load_eval_dataset("data/evaluation/test_queries.yaml")
evaluator = RAGASEvaluator(pipeline)
result = evaluator.evaluate(samples)
print_console_report(result)
```

## 运行方式

### 构建索引

```bash
# 原始实现
uv run python -m src.preprocess.indexer

# LangChain 实现
uv run python -m src_langchain.preprocess.indexer
```

### 启动 Web UI

```bash
# 原始实现
uv run python app.py

# LangChain 实现
uv run python app.py --backend langchain
```

### 运行评估

```bash
# 原始实现
uv run python scripts/evaluate.py --backend src --limit 5

# LangChain 实现
uv run python scripts/evaluate.py --backend langchain --limit 5
```

## LangChain 组件使用汇总

| 组件 | 来源 | 用途 |
|------|------|------|
| `DirectoryLoader` | `langchain_community` | 批量加载食谱 .md 文件 |
| `TextLoader` | `langchain_community` | 加载单个文件 |
| `MarkdownHeaderTextSplitter` | `langchain_text_splitters` | 按 H1/H2 切分文档 |
| `HuggingFaceEmbeddings` | `langchain_huggingface` | 中文文本向量化 |
| `FAISS` | `langchain_community.vectorstores` | 稠密向量存储与检索 |
| `BM25Retriever` | `langchain_community.retrievers` | 稀疏关键词检索 |
| `RRFFusionRetriever` | **自定义 `BaseRetriever`** | RRF 混合检索 |
| `ChatOpenAI` | `langchain_openai` | LLM 调用 (DeepSeek API) |
| `ChatPromptTemplate` | `langchain_core.prompts` | 系统/用户提示词模板 |
| `StrOutputParser` | `langchain_core.output_parsers` | LLM 输出解析 |
| `with_structured_output()` | `langchain_openai` | 意图分类的结构化 JSON 输出 |
| `BaseRetriever` | `langchain_core.retrievers` | 自定义检索器基类 |
| `Document` | `langchain_core.documents` | 文档数据结构 |

## 关键技术决策

1. **RRF 融合需自定义 Retriever** — LangChain 内置 `EnsembleRetriever` 仅支持加权和，自定义 `RRFFusionRetriever(BaseRetriever)` 约 60 行代码实现 RRF 融合。

2. **元数据提取保留自定义 Transformer** — `MarkdownHeaderTextSplitter` 不提取 difficulty/calories，通过 `RecipeDocumentTransformer` 在分块后附加。

3. **jieba 分词共享** — `chinese_tokenize()` 提取至 `shared/tokenizer.py`，消除 `src/preprocess/config.py` 和 `src_langchain/retrieval/tokenizer.py` 的重复定义。

4. **评估模块共享** — RAGAS 评估器通过 pipeline 注入实现后端无关，`src/evaluation/` 和 `src_langchain/evaluation/` 合并为 `shared/evaluation/`。

5. **结构化输出用 `with_structured_output`** — 替代 v1.x 手写的 JSON mode + `json.loads()`。

6. **Pipeline 保持过程式编排** — 不使用完整的 LCEL 链，因为检索策略选择涉及复杂的条件分支和多通道检索，过程式代码更清晰可维护。

## LangChain 版本兼容

当前 LangChain 版本中的 API 变更及适配：

| 旧 API | 新 API | 影响组件 |
|--------|--------|----------|
| `BaseRetriever.get_relevant_documents()` | `invoke()` | ensemble.py, sparse_retriever.py, app.py |
| `MarkdownHeaderTextSplitter.create_documents()` | `split_text()` | splitter.py |
| `BM25Retriever.get_scores()` | 已移除，直接调用 `invoke()` | sparse_retriever.py |
