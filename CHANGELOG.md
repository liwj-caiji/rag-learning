# Changelog

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
