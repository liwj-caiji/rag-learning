# Changelog

## 2026-05-28 — 项目结构优化：共享模块提取 & 双后端统一

### 项目结构重组

#### 1. 共享包 `shared/` 提取
- **`shared/evaluation/`**：RAGAS 评估模块从 `src/evaluation/` 提取，通过 pipeline 构造器注入消除对 `src/` 或 `src_langchain/` 的硬依赖
- **`shared/tokenizer.py`**：jieba 中文分词函数 `chinese_tokenize()` 从此前的 4 处重复定义统一为单一源
- **删除** `src/evaluation/`（5 个文件）和 `src_langchain/evaluation/`（5 个文件）

#### 2. 消除命名问题
- `src-langchain/rag_langchain/` → `src_langchain/`（移除短横线，消除多余嵌套层级）
- 删除空的 `src-langchain/` 目录
- 所有 `from rag_langchain` 导入 → `from src_langchain`

#### 3. 统一入口
- `app.py` 合并两份实现（原 420+420 行 → 340 行），通过 `--backend src|langchain` 切换
- `scripts/evaluate.py` 新增 `--backend src|langchain` 参数

#### 4. LangChain 后端索引隔离
- LangChain 索引用独立文件名：`chunks_langchain.pkl`, `bm25_langchain.pkl`, `faiss_langchain/`
- 修复 `BASE_DIR` 路径计算（3 层 → 2 层 dirname）

### Bug 修复

- 修复 LangChain API 兼容性：`get_relevant_documents()` → `invoke()`
- 修复 `MarkdownHeaderTextSplitter.create_documents()` → `split_text()`
- 修复 `BM25Retriever.get_scores()` 已移除的问题
- 修复 `src_langchain/config.py` BASE_DIR 路径错误

### 评估验证

- 双后端小规模评估对比（3 样本）：`context_precision`、`context_recall`、`answer_relevancy` 完全一致，`faithfulness` 和 `answer_correctness` 在正常浮动范围内

### 涉及文件

| 文件 | 改动类型 |
|------|----------|
| `shared/` | 新增（7 个文件） |
| `src_langchain/` | 重组（从 `src-langchain/rag_langchain/` 迁移） |
| `src/evaluation/` | 删除（迁移至 `shared/evaluation/`） |
| `src_langchain/evaluation/` | 删除（迁移至 `shared/evaluation/`） |
| `src_langchain/retrieval/tokenizer.py` | 删除（迁移至 `shared/tokenizer.py`） |
| `src_langchain/app.py` | 删除（合并至 `app.py`） |
| `app.py` | 重写（统一双后端入口） |
| `scripts/evaluate.py` | 修改（新增 --backend 参数） |
| `src/preprocess/config.py` | 修改（chinese_tokenize 改为 re-export） |
| `tests/test_evaluation.py` | 修改（导入路径迁移至 shared） |
| `.gitignore` | 修改（新增 logs/） |
| `docs/功能实现说明_LangChain重构.md` | 更新（项目结构、API 兼容说明） |
| `docs/CHANGELOG.md` | 修改（新增本条） |

## 2026-05-27 — RAGAS 评估：ground truth 对齐 & 中文适配完成

### 数据修复

#### 评估数据集 ground truth 重写
- **文件**: `data/evaluation/test_queries.yaml`
- 所有 ground truth 与 `base/HowToCook` 中的实际食谱文件对齐（如麻婆豆腐使用咸鸭蛋版、水煮鱼使用巴沙鱼版等）
- 知识库不存在的菜品（番茄炒蛋、饺子）移除 ground truth，context_recall 正确跳过
- **效果**：context_recall 0.250→0.765（+206%），context_precision 0.267→0.442（+66%），howto 做法查询 context_recall 0.200→0.875

### 文档更新

#### `docs/evaluation.html`
- 整体指标卡片更新为最新评分
- 按意图分组表格更新
- 副标题补充 ground truth 对齐说明

### 涉及文件

| 文件 | 改动类型 |
|------|----------|
| `data/evaluation/test_queries.yaml` | 修改（ground truth 重写） |
| `docs/evaluation.html` | 修改（分数更新） |
| `docs/CHANGELOG.md` | 修改（新增本条） |

## 2026-05-27 — RAGAS 中文评估优化 & 评估模块上线

### 功能新增

#### 1. RAGAS 评估模块
- **新增 `src/evaluation/`**：完整的 RAGAS 评估流水线
  - `evaluator.py`：核心评估器，封装 RAGAS metrics 计算
  - `dataset.py`：从 YAML 加载测试查询，构建评估数据集
  - `reporter.py`：生成 HTML 评估报告
  - `config.py`：评估专用配置（LLM、指标、批处理）
- **新增 `data/evaluation/test_queries.yaml`**：30 条中文烹饪领域测试查询，覆盖做法查询、原料查询、推荐、事实查询四类意图
- **新增 `scripts/evaluate.py`**：一键运行评估并生成报告
- **新增 `scripts/generate_report.py`**：独立 HTML 报告生成

#### 2. RAGAS 中文提示词适配（adapt_prompts）
- **文件**: `src/evaluation/evaluator.py`
- 使用 ragas 官方 `adapt_prompts(language="chinese")` API 自动翻译所有指标的提示词为中文
- 首次运行通过 deepseek-chat 翻译并 pickle 缓存至 `.ragas_cache/`，后续运行零 API 调用
- 替换了早期手写 PydanticPrompt 方案（`prompts.py`，已删除），消除 398 行需与 ragas 内部 schema 精确对齐的维护负担

#### 3. Pipeline trace 增强
- **文件**: `src/generation/pipeline.py`
- `trace()` 方法返回的 chunk 中新增 `text` 字段，供评估模块提取上下文文本

### 涉及文件

| 文件 | 改动类型 |
|------|----------|
| `src/evaluation/__init__.py` | 新增 |
| `src/evaluation/evaluator.py` | 新增 |
| `src/evaluation/dataset.py` | 新增 |
| `src/evaluation/reporter.py` | 新增 |
| `src/evaluation/config.py` | 新增 |
| `src/evaluation/prompts.py` | 已删除（被 adapt_prompts 替代） |
| `data/evaluation/test_queries.yaml` | 新增 |
| `docs/evaluation.html` | 新增（评估报告） |
| `scripts/evaluate.py` | 新增 |
| `scripts/generate_report.py` | 新增 |
| `src/generation/pipeline.py` | 修改（trace 增加 text 字段） |
| `requirements.txt` | 修改（新增 ragas, datasets, pyyaml 等依赖） |
| `.gitignore` | 修改（新增 .ragas_cache/） |
| `docs/CHANGELOG.md` | 修改（新增本条） |

## 2026-05-26 — Mermaid 示意图布局优化 & 功能增强

### 功能增强

#### 1. 查询意图类别关键词英文化
- **文件**: `src/rewriting/intent.py`
- 将 `_CATEGORY_KEYWORDS` 字典键从中文（"素菜"/"肉菜"/"汤"等）改为英文（"vegetable_dish"/"meat_dish"/"soup"等），避免内部编码问题

#### 2. 标题分组精细控制
- **文件**: `src/preprocess/splitter.py`
- `_group_by_headings()` 新增 `split_level` 参数，控制按几级标题切分，修复 L3 子章节拆分逻辑

#### 3. 完整食谱上下文注入
- **文件**: `src/generation/pipeline.py` / `src/generation/llm_generator.py` / `src/generation/template.py`
- 当检索到目标菜品时，自动读取原始 `.md` 源文件，将完整食谱内容作为 "完整食谱" 条目前置到 LLM 上下文
- LLM prompt 增加对「完整食谱」条目的使用指示
- 未精确命中时生成"不知道"+推荐相似菜品（Template 和 LLM 双模式）

### Mermaid 示意图布局优化

#### 1. 整体布局调整
- 所有 diagram 去除容器宽度限制（`max-width: 100%`），SVG 按自然大小渲染
- `useMaxWidth: false` 取消 Mermaid 自动缩放，保留原始分辨率
- 节点文字极度精简，减少子图内部空白
- `overflow-x: auto` 支持横向滚动查看超宽部分

#### 2. `docs/retrieval.html`
- 重组子图：CACHE → DENSE+SPARSE → FUSION → RECOMMEND 纵向排列
- MMR 细节作为侧边子图附着
- 配置引用精简为一行标注

#### 3. `docs/rewriting.html`
- LLM 路径与规则路径双子图重新排列，合并冗余节点
- 回退路径使用虚线箭头标注

#### 4. `docs/generation.html`
- 移除两列布局，RAGPipeline 与系统架构图上下排列各占满宽
- 文字大幅精简，子图内边距压缩

### 涉及文件

| 文件 | 改动类型 |
|------|----------|
| `docs/retrieval.html` | 修改（布局优化） |
| `docs/rewriting.html` | 修改（布局优化） |
| `docs/generation.html` | 修改（布局优化） |
| `docs/CHANGELOG.md` | 修改（新增本条） |
| `CHANGELOG.md` | 删除（迁移至 docs/） |
| `src/preprocess/splitter.py` | 修改（标题分组精细控制） |
| `src/rewriting/intent.py` | 修改（类别关键词英文化） |
| `src/generation/pipeline.py` | 修改（完整食谱上下文注入） |
| `src/generation/llm_generator.py` | 修改（prompt 优化） |
| `src/generation/template.py` | 修改（"不知道"+推荐） |

## 2026-05-25 — UI 卡死修复 & 配置集中管理

### 问题修复

#### 1. Gradio UI 点击按钮卡死
- **根因**：`demo.load` 触发 `_load_stats`，内部调用 `collect_all_recipes()` 扫描 363 个食谱文件耗时约 8 秒，期间 Gradio 事件队列被完全占用，所有按钮点击被排队但无反馈，UI 看起来卡死
- **修复**：移除 `demo.load` 触发的 stats 加载，改为仅用户点击"刷新"时加载
- **补充**：stats 结果加入内存缓存，避免重复扫描磁盘

#### 2. 操作无反馈
- **根因**：所有事件缺少 `show_progress` 参数，用户无法感知"排队中"或"处理中"状态
- **修复**：所有耗时事件（Pipeline 执行、检索、对话、数据刷新）添加 `show_progress="full"`

#### 3. 事件队列配置不完整
- **根因**：部分事件缺少 `concurrency_limit`，且队列未设 `max_size`，极端情况下请求无限堆积
- **修复**：统一所有事件的 `concurrency_limit`，队列增加 `max_size=20`

#### 4. `rewriter.py` 模型名残留
- **根因**：`LLMQueryRewriter` 的默认模型为 `"claude-sonnet-4-20250506"`，与项目其余模块使用的 `"deepseek-v4-flash"` 不一致
- **修复**：改为引用 `src.config.LLM_INTENT_MODEL`

### 架构优化

#### 1. 配置集中管理
- **新增 `src/config.py`** 作为全局唯一配置源，按功能划分为 9 个配置段
- **消除重复定义**：此前 `FAISS_INDEX_PATH` / `CHUNKS_PATH` / `BM25_INDEX_PATH` 在 `indexer.py` 和 `hybrid.py` 中各定义一次，现统一从 `src.config` 导入
- **涉及模块**：`preprocess` / `retrieval` / `rewriting` / `generation` / `app`，共 10 个文件
- **`src/preprocess/config.py`** 缩编为向后兼容的 re-export 层 + `chinese_tokenize` 函数

#### 2. 检索候选数优化
调小各环节候选数量以降低 RRF 融合与排序开销：

| 配置项 | 原值 | 新值 |
|--------|------|------|
| `DENSE_CANDIDATES_K` | 50 | 20 |
| `SPARSE_CANDIDATES_K` | 50 | 20 |
| `RECOMMEND_PROBE_CANDIDATES` | 30 | 15 |
| `PIPELINE_HOWTO_K` | 50 | 20 |
| `PIPELINE_HOWTO_DENSE_K` / `SPARSE_K` | 100 | 50 |
| `PIPELINE_INGREDIENT_K` | 50 | 20 |
| `PIPELINE_INGREDIENT_DENSE_K` / `SPARSE_K` | 100 | 50 |

### 涉及文件

| 文件 | 改动类型 |
|------|----------|
| `src/config.py` | 新增 |
| `app.py` | 修改（UI 卡死修复、队列配置、引用 config） |
| `src/preprocess/config.py` | 修改（缩编为 re-export） |
| `src/preprocess/indexer.py` | 修改（导入迁移至 src.config） |
| `src/retrieval/hybrid.py` | 修改（导入及默认值迁移） |
| `src/retrieval/diversity.py` | 修改（导入及默认值迁移） |
| `src/retrieval/filters.py` | 修改（导入迁移） |
| `src/rewriting/llm_intent.py` | 修改（导入及默认值迁移） |
| `src/rewriting/rewriter.py` | 修改（模型名修复） |
| `src/generation/llm_generator.py` | 修改（导入及默认值迁移） |
| `src/generation/pipeline.py` | 修改（导入及默认值迁移） |
