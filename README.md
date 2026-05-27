# 🍳 食谱 RAG 系统

基于 **检索增强生成（RAG）** 架构的智能食谱问答系统。  
数据源为 [HowToCook](https://github.com/Anduin2017/HowToCook) 开源食谱库，覆盖 1000+ 道中式家常菜。

---

## 功能特性

| 功能 | 说明 |
|------|------|
| **层级分块** | 将 Markdown 食谱按 dish / section / subsection 三级结构分块 |
| **混合检索** | 稠密检索（FAISS + text2vec）与稀疏检索（BM25 + jieba）融合 |
| **RRF 融合** | Reciprocal Rank Fusion 融合两路检索结果 |
| **查询改写** | 规则驱动 + LLM（deepseek-v4-flash）双模式意图识别与改写 |
| **多探针搜索** | 推荐场景下生成多个搜索探针提升召回多样性 |
| **MMR 重排序** | Maximum Marginal Relevance 平衡相关性与多样性 |
| **元数据过滤** | 按难度、类别、卡路里等条件过滤结果 |
| **LLM 生成** | 模板渲染 + LLM（deepseek-v4-flash）双模式答案生成 |
| **RAGAS 评估** | 5 项指标评估（回答相关性、上下文精确度/召回率、忠实度、正确性） |
| **Web UI** | Gradio 交互界面，支持 Pipeline 可视化与对话 |

---

## 快速开始

### 环境要求

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip

### 1. 克隆仓库

```bash
git clone https://github.com/liwj-caiji/rag-learning.git
cd rag-learning
```

### 2. 创建虚拟环境并安装依赖

**使用 uv（推荐）：**

```bash
uv venv
source .venv/bin/activate    # Linux / macOS
.venv\Scripts\activate       # Windows PowerShell
.venv\Scripts\activate.bat   # Windows CMD

uv pip install -r requirements.txt
```

**使用 pip：**

```bash
python -m venv .venv
source .venv/bin/activate    # Linux / macOS
.venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

首次安装会自动下载 sentence-transformers 模型（约 400MB），
以及 `faiss-cpu`、`jieba` 等依赖。

### 3. 构建索引

```bash
python -m src.preprocess.indexer
```

遍历 `base/HowToCook/dishes/` 下所有食谱文件，执行分块、向量化并构建 FAISS + BM25 索引。  
输出保存在 `data/vectorstore/`。

### 4. 启动 Web UI

```bash
python app.py
```

浏览器访问 `http://127.0.0.1:7860`

### 5.（可选）配置 LLM

设置环境变量启用 LLM 驱动的意图识别与答案生成：

```bash
# Windows CMD
set DEEPSEEK_API_KEY=your-api-key

# Windows PowerShell
$env:DEEPSEEK_API_KEY="your-api-key"

# Linux / macOS
export DEEPSEEK_API_KEY=your-api-key
```

然后在 Web UI 中勾选 **LLM 模式** 即可切换。

---

## 使用方式

### Web UI

启动后浏览器打开 `http://127.0.0.1:7860`，界面包含：

| 标签页 | 功能 |
|--------|------|
| **📊 Pipeline 总览** | 输入查询，展示改写 → 检索 → 生成的完整流程 |
| **🔍 检索演示** | 单独测试混合/稠密/稀疏检索，对比召回效果 |
| **📦 数据概览** | 数据集统计（食谱数、品类分布、分块数） |
| **💬 对话助手** | 侧栏对话窗口，支持连续提问 |

### RAGAS 评估

运行 RAGAS 评估流水线，用 5 项标准指标衡量检索和生成质量：

```bash
# 完整评估（30 样本，LLM 模式）
uv run python scripts/evaluate.py --mode llm --limit 30

# 仅评估指定意图
uv run python scripts/evaluate.py --mode llm --intent howto

# 输出 JSON 报告
uv run python scripts/evaluate.py --mode llm --output report.json
```

评估指标说明：

| 指标 | 含义 | 当前值 |
|------|------|:---:|
| **answer_relevancy** | 回答与问题的相关程度 | 0.712 |
| **context_precision** | 检索到的上下文中有多少与问题相关 | 0.442 |
| **context_recall** | 回答中可归因于上下文的比例 | 0.765 |
| **faithfulness** | 回答是否忠于检索到的上下文（无幻觉） | 0.817 |

> 评估使用 `ragas adapt_prompts(language="chinese")` 自动将指标提示词翻译为中文，首次运行缓存至 `.ragas_cache/`。

### Python API

```python
from src.generation import RAGPipeline

# 规则模式（无外部依赖）
pipeline = RAGPipeline()
answer = pipeline.run("麻婆豆腐怎么做")
print(answer)

# LLM 模式（需设置 DEEPSEEK_API_KEY）
llm_pipeline = RAGPipeline(use_llm=True)
answer = llm_pipeline.run("今天有什么清淡的推荐？")

# 获取完整中间状态
trace = pipeline.trace("红烧肉怎么做")
print(trace["intent"], trace["filters"], trace["chunks"])
```

```python
from src.retrieval import hybrid_search, recommend_dishes

# 混合搜索
results = hybrid_search("红烧肉 做法", k=5)
for r in results:
    print(r["chunk"]["metadata"]["dish_name"], r["score"])

# 菜品推荐
recommendations = recommend_dishes(
    query="今天想吃点清淡的",
    k=3,
    filters={"calories": "low"},
)
```

```python
from src.rewriting import LLMIntentClassifier

# 仅使用 LLM 意图识别
classifier = LLMIntentClassifier()
result = classifier.classify("有没有适合新手的简单肉菜？")
print(result.intent, result.filters, result.probes)
```

---

## 项目结构

```
rag-learning/
├── app.py                       # Gradio Web UI 入口
├── requirements.txt             # Python 依赖
│
├── base/HowToCook/              # 食谱数据集（git submodule）
│   └── dishes/                  #   Markdown 食谱文件
│       ├── meat_dish/           #     肉菜
│       ├── vegetable_dish/      #     素菜
│       ├── soup/                #     汤羹
│       ├── aquatic/             #     水产
│       ├── breakfast/           #     早餐
│       ├── staple/              #     主食
│       ├── dessert/             #     甜点
│       └── drink/               #     饮品
│
├── data/
│   ├── vectorstore/               # 索引数据（构建后生成）
│   │   ├── faiss.index            #   FAISS 稠密向量索引
│   │   ├── chunks.pkl             #   分块元数据
│   │   └── bm25_index.pkl         #   BM25 稀疏索引
│   └── evaluation/
│       └── test_queries.yaml      # 评估数据集（30 条中文查询 + ground_truth）
│
├── scripts/
│   ├── evaluate.py                # RAGAS 评估入口
│   └── generate_report.py         # HTML 评估报告生成
│
├── tests/                         # 单元测试
│   ├── test_pipeline.py
│   ├── test_evaluation.py
│   └── ...
│
├── docs/
│   ├── CHANGELOG.md               # 变更日志
│   ├── evaluation.html            # RAGAS 评估报告
│   ├── retrieval.html             # 检索模块架构图
│   ├── rewriting.html             # 改写模块架构图
│   ├── generation.html            # 生成模块架构图
│   └── preprocess.html            # 预处理模块架构图
│
└── src/
    ├── preprocess/              # 数据预处理
    │   ├── config.py            #   配置（路径、模型、分词）
    │   ├── splitter.py          #   RecipeSplitter 层级分块
    │   └── indexer.py           #   索引构建（FAISS + BM25）
    │
    ├── retrieval/               # 检索模块
    │   ├── hybrid.py            #   混合检索 + RRF + 推荐
    │   ├── filters.py           #   元数据过滤
    │   └── diversity.py         #   MMR + 类别轮询
    │
    ├── rewriting/               # 查询改写
    │   ├── intent.py            #   规则驱动意图分类
    │   ├── rewriter.py          #   改写器抽象
    │   └── llm_intent.py        #   LLM 意图分类器
    │
    ├── generation/              # 答案生成
    │   ├── base.py              #   Generator 抽象基类
    │   ├── template.py          #   模板生成器
    │   ├── llm_generator.py     #   LLM 生成器
    │   └── pipeline.py          #   RAGPipeline 端到端编排
    │
    └── evaluation/              # RAGAS 评估
        ├── evaluator.py         #   RAGASEvaluator 核心评估器
        ├── dataset.py           #   评估数据集加载与过滤
        ├── reporter.py          #   HTML / JSON 报告生成
        └── config.py            #   评估配置（LLM、指标、批处理）
```

---

## 技术架构

```
用户查询
    │
    ▼
┌─────────────────┐
│  查询改写         │  ← 规则匹配 / LLMIntentClassifier
│  意图识别         │     (deepseek-v4-flash)
│  约束提取         │
│  多探针生成       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  混合检索         │
│  ┌─────┐ ┌────┐ │
│  │FAISS│ │BM25│ │  ← RRF 融合 (k=60)
│  └──┬──┘ └──┬─┘ │
│     └──┬────┘   │
│   ┌────▼────┐   │
│   │MMR 重排  │   │  ← 按需启用多样性排序
│   └─────────┘   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  生成回答         │  ← 模板渲染 / LLMGenerator
│                   │     (deepseek-v4-flash)
└─────────────────┘
```

### 核心依赖

| 组件 | 选型 |
|------|------|
| 稠密检索 | FAISS + `shibing624/text2vec-base-chinese` |
| 稀疏检索 | BM25Okapi + jieba |
| 融合策略 | RRF（Reciprocal Rank Fusion） |
| 重排序 | MMR（Maximum Marginal Relevance） |
| 意图识别 | 规则 / deepseek-v4-flash |
| 答案生成 | 模板 / deepseek-v4-flash |
| Web UI | Gradio |
| LLM 客户端 | OpenAI 兼容 SDK |

---

## 配置说明

### LLM API

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | — |
| `LLM_MODEL` | 模型名称（代码中可配） | `deepseek-v4-flash` |
| `LLM_API_BASE` | API 地址（代码中可配） | `https://api.deepseek.com` |

### 分块参数（`src/preprocess/config.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `REMOVE_FOOTER` | `True` | 移除 GitHub PR 脚注 |
| `SPLIT_H3` | `True` | 将 `## 操作` 下的 `###` 拆为独立子块 |
| `EMBED_MODEL` | `shibing624/text2vec-base-chinese` | 嵌入模型 |

---

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.2 | 2026-05-27 | RAGAS 评估模块 + 中文提示词适配（adapt_prompts） + 评估数据集 ground truth 对齐 |
| v1.1 | 2026-05-25 | LLM 集成 + Gradio Web UI |
| v1.0 | 2026-05-25 | 基础 RAG 流水线（预处理、检索、改写、生成） |
