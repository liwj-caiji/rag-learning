# 食谱 RAG 智能问答系统

基于混合检索 + LLM 生成的中文食谱 RAG 问答系统，363+ 道菜谱知识库（[HowToCook](https://github.com/Anduin2017/HowToCook)）。

## 项目概览

- **用途**：用户输入烹饪查询（推荐、做法、原料、知识），系统经查询改写 → 混合检索 → 上下文增强 → 答案生成返回回答
- **技术栈**：Python · FAISS · BM25 · RRF · LangChain · DeepSeek · Gradio · RAGAS · Langfuse
- **依赖管理**：uv（`.venv/` 虚拟环境，`pyproject.toml` 声明依赖）
- **知识库**：`base/HowToCook/dishes/` 下 363+ 个 Markdown 菜谱文件

### uv 环境管理

```bash
uv sync                  # 安装/同步依赖
uv add <package>         # 添加新依赖
uv run python <script>   # 在项目 venv 中运行脚本
```

无需手动激活环境，直接使用 `uv run` 即可。

## 架构

### 双后端架构

| 后端 | 入口 | 特点 |
|------|------|------|
| 原生 SDK | `src/` | 手动实现 FAISS/BM25/RRF/OpenAI SDK，理解底层机制 |
| LangChain | `src_langchain/` | LangChain 组件封装，对接 Langfuse 可观测性 |

通过 `app.py --backend src|langchain` 切换，内部通过 `_resolve(module_name)` 动态导入。

### Pipeline 四阶段

1. **改写** — QueryRewriter / LLMIntentClassifier：意图分类（4 类）+ 约束提取 + 查询改写
2. **检索** — HybridRetriever：FAISS 稠密 + BM25 稀疏 → RRF 融合，意图驱动差异化路由
3. **增强** — `_enrich_with_full_recipe()`：howto/ingredient 场景注入完整食谱源文件，避免 LLM 因分块缺失原料而幻觉
4. **生成** — TemplateGenerator（规则）/ LLMGenerator（DeepSeek API，异常时回退模板）

### 意图类型

- **recommendation**：多探针搜索 + MMR 多样性重排 + 类别轮询
- **howto**：偏重「操作」章节，检测目标菜名后注入完整食谱
- **ingredient**：偏重「必备原料和工具」章节
- **factual**：通用混合检索

### 共享模块

`shared/` 包中 jieba 分词器（`shared/tokenizer.py`）和 RAGAS 评估模块（`shared/evaluation/`）被双后端共享。

## 关键实现细节

### 层级 Markdown 分块

`src/preprocess/splitter.py:RecipeSplitter` — 每个 `.md` 产出 L1-dish / L2-section / L3-subsection 三级 chunk，从导言正则提取难度和卡路里。

### 混合检索

- 稠密：SentenceTransformer (`shibing624/text2vec-base-chinese`, 768d) → FAISS IndexFlatIP（L2 归一化后用内积等价余弦相似度）
- 稀疏：jieba 分词 → BM25Okapi（`rank_bm25`）
- 融合：RRF k=60，经典值，消除两路分数量纲差异

### 配置

`src/config.py` 为全局唯一配置源，9 个配置段：Paths, Embedding, Chunking, Retrieval, Diversity, LLM Intent, LLM Gen, App, Logging。

### 评估

30 条中文测试查询（`data/evaluation/test_queries.yaml`），RAGAS 5 指标（context_precision/recall, faithfulness, relevancy, correctness），`adapt_prompts(language="chinese")` 适配中文。

### 可观测性

仅 LangChain 后端：Langfuse `@observe` 装饰器 + `LangchainCallbackHandler`，通过 `callbacks` 参数在 Pipeline → IntentClassifier → LLMGenerator 间完整传递。

## 常用命令

```bash
# 索引构建
uv run python -m src.preprocess.indexer          # 原生后端
uv run python -m src_langchain.preprocess.indexer  # LangChain 后端

# 启动 Web UI
uv run python app.py                              # 默认 src 后端
uv run python app.py --backend langchain          # LangChain 后端

# 评估
uv run python scripts/evaluate.py --backend src
uv run python scripts/evaluate.py --backend langchain
```

## 已知限制

- 食谱数据为静态 Markdown，不支持增量更新
- 混合检索相似度启发式（MMR 回退路径）为近似值
- 无流式输出，回答需等待完整生成
- 仅支持 DeepSeek 作为 LLM provider（未适配 Anthropic / Ollama）
- LangChain 后端与原生后端功能非 100% 等价


## gstack 角色路由
- 当需要产品决策、范围判断时，使用 /office-hours 或 /plan-ceo-review
- 当需要架构审查时，使用 /plan-eng-review
- 当代码准备合并前，使用 /review 进行代码审查
- 当需要端到端测试时，使用 /qa
- 当准备发布时，使用 /ship