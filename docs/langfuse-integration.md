# Langfuse 集成说明

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        app.py (Gradio)                       │
│                                                              │
│  pipeline.run(query)  ──── 返回 answer 字符串                 │
│  pipeline.trace(query) ──── 返回完整 dict（含 chunks、intent） │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                  RAGPipeline (pipeline.py)                    │
│                                                              │
│  @observe("RAGPipeline.run")   ← Langfuse 装饰器，自动创建    │
│  def run(query, top_k):         Trace（包含 input/output）    │
│      rewriter.classify(query)  ────────────────┐             │
│      _retrieve(intent_result, top_k, query)    │             │
│      generator.generate(...)                   │             │
│      _enrich_trace(...)       ← 注入 metadata │             │
│      return answer                             │             │
│                                                ▼             │
│  _enrich_trace():                     tracing.py              │
│    lf.update_current_span(            ─────────               │
│      metadata={                     get_langfuse_handler()   │
│        intent: ...,        ───────────────┘                   │
│        target_dish: ...,   LangchainCallbackHandler           │
│        num_chunks: ...,    作为 callbacks 传入 LLM 链，        │
│        contexts: [...],    自动捕获 LLM 调用的 token 用量      │
│        ...                                                    │
│      }                                                       │
│    )                                                         │
└──────────────────────────────────────────────────────────────┘
```

## 2. 文件说明

### 2.1 `src_langchain/config.py` — 环境变量配置

```python
# Langfuse 通过这两个环境变量自动初始化，无需代码中硬编码密钥
LANGFUSE_PUBLIC_KEY_ENV = "LANGFUSE_PUBLIC_KEY"   # 从 langfuse.com 项目设置获取
LANGFUSE_SECRET_KEY_ENV = "LANGFUSE_SECRET_KEY"   # 从 langfuse.com 项目设置获取
LANGFUSE_BASE_URL_ENV = "LANGFUSE_BASE_URL"       # 自托管时使用，SaaS 默认为 cloud.langfuse.com
```

用户只需在 shell 中设置：
```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-xxx
export LANGFUSE_SECRET_KEY=sk-lf-xxx
# LANGFUSE_BASE_URL 可选，默认 https://cloud.langfuse.com
```

### 2.2 `src_langchain/tracing.py` — CallbackHandler 初始化

**作用**：创建全局共享的 `LangchainCallbackHandler`，捕获 LangChain 链中每一步的 LLM 调用（token 用量、耗时、输入输出）。

**关键设计**：
- **惰性初始化 + 全局复用**：`_handler` 是模块级单例，首次调用时创建，后续复用
- **安全降级**：如果 `langfuse` 包未安装或密钥未配置，静默返回 `None`，不影响业务逻辑
- **与 `@observe` 协作**：`CallbackHandler` 自动读取当前 trace 上下文，将 LLM 调用作为子 span 挂到 `@observe` 创建的 trace 下

```python
def get_langfuse_handler() -> Optional[object]:
    global _handler, _checked
    if _checked:
        return _handler       # 已初始化过，直接返回缓存

    _checked = True
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        log.info("Langfuse not configured")
        return None

    from langfuse.langchain import CallbackHandler
    _handler = CallbackHandler()
    return _handler
```

### 2.3 `src_langchain/pipeline.py` — 核心集成点

`RAGPipeline` 是 Langfuse 集成的主入口，做了三件事：

#### (a) `@observe` 装饰器自动创建 Trace

```python
from langfuse import observe  # 如果未安装则降级为无操作装饰器

@observe(name="RAGPipeline.run")
def run(self, query: str, top_k: int = 5) -> str:
    # 函数参数自动成为 trace.input = {"query": "...", "top_k": 5}
    # 函数返回值自动成为 trace.output = "回答文本"
    ...
    return answer
```

#### (b) `_enrich_trace()` 注入元数据

每轮查询结束后，通过 `lf.update_current_span()` 将以下信息写入 trace：

| 字段 | 示例值 | 用途 |
|------|--------|------|
| `name` | `"RAG: 红烧肉怎么做"` | Dashboard 中易于识别 |
| `metadata.intent` | `"howto"` | 按意图分析质量 |
| `metadata.target_dish` | `"红烧肉"` | 按菜品分析命中率 |
| `metadata.rewritten_query` | `"红烧肉 操作"` | 改写后的检索查询 |
| `metadata.num_chunks` | `5` | 检索到的 chunk 数 |
| `metadata.total_elapsed_s` | `14.18` | 总耗时（秒），用于性能监控 |
| `metadata.model` | `"deepseek-v4-flash"` | 按模型对比质量 |
| `metadata.use_llm` | `false` | 规则 vs LLM 模式对比 |
| `metadata.filters` | `{}` | 用户约束条件（热量、难度等） |
| `metadata.probes` | `["红烧肉 操作"]` | 多路搜索探针 |
| `metadata.contexts` | `["步骤1...", "原料..."]` | 检索到的上下文全文 |
| `metadata.tag_intent` | `"howto"` | 用于 Dashboard 筛选（langfuse v4.x span API 不支持 tags，以 metadata 替代） |
| `metadata.tag_model` | `"deepseek-v4-flash"` | 用于 Dashboard 筛选 |
| `metadata.tag_backend` | `"langchain"` | 用于 Dashboard 筛选 |
| `metadata.tag_mode` | `"template"` | llm / template 模式筛选 |

**为什么要用 `update_current_span` 而不是 `update_current_trace`**

langfuse v4.7.0 使用 OpenTelemetry，`@observe` 装饰器创建的是一个 **span**（根 span 即 trace）。`Langfuse` 客户端暴露的 API 是 `update_current_span()`，不存在 `update_current_trace()` 方法。`update_current_span` 也不支持 `tags` 参数，因此标签信息以 `tag_` 前缀放入 metadata。

#### (c) callback 传递链

```python
# RAGPipeline.__init__:
self._langfuse_handler = get_langfuse_handler()   # 获取全局 handler
_callbacks = [handler] if handler else None

self.rewriter = get_intent_classifier(..., callbacks=_callbacks)
self.generator = LLMGenerator(..., callbacks=_callbacks)
```

callback 被传入 `LLMIntentClassifier` 和 `LLMGenerator`，在调用 `ChatOpenAI` 时作为 `config["callbacks"]` 传入：

```python
# llm_classifier.py — 意图识别 LLM 调用
config = {}
if self.callbacks:
    config["callbacks"] = self.callbacks
structured_llm.invoke(prompt, config=config)

# llm_chain.py — 生成 LLM 调用
invoke_config = {}
if self.callbacks:
    invoke_config["callbacks"] = self.callbacks
chain.invoke({"user_prompt": user_prompt}, config=invoke_config)
```

这使得 Langfuse 自动捕获每次 LLM 调用的 token 用量、模型名、延迟，作为子 span 出现在 trace 中。

## 3. Callback 传递链路

```
RAGPipeline.__init__()
  │
  ├─ get_langfuse_handler()           ← 只调一次，全局复用
  │     └─ LangchainCallbackHandler   ← langfuse.langchain 提供
  │
  ├─ rewriter (LLMIntentClassifier)
  │     .callbacks = [handler]         ← 意图识别 LLM 调用时传入
  │     └─ ChatOpenAI.invoke(..., config={"callbacks": [handler]})
  │           → Langfuse 自动创建 "ChatOpenAI" 子 span
  │             记录 model、tokens、latency、input/output
  │
  └─ generator (LLMGenerator)
        .callbacks = [handler]         ← 生成 LLM 调用时传入
        └─ chain.invoke(..., config={"callbacks": [handler]})
              → Langfuse 自动创建第二个 "ChatOpenAI" 子 span
```

**效果**：在 Langfuse Dashboard 中，每条 trace 展开后能看到完整的 span 树：

```
Trace: "RAG: 红烧肉怎么做"
├── Span: RAGPipeline.run (总耗时)
│   ├── Span: ChatOpenAI (意图识别 LLM 调用 — deepseek-v4-flash)
│   │      tokens: 150 input, 50 output
│   │      latency: 1.2s
│   │
│   ├── (检索步 — 无 LLM，metadata 中有 contexts)
│   │
│   └── Span: ChatOpenAI (答案生成 LLM 调用 — deepseek-v4-flash)
│          tokens: 800 input, 300 output
│          latency: 3.5s
```

## 4. 数据流

```
用户输入 "红烧肉怎么做"
  │
  ▼
@observe("RAGPipeline.run")  →  创建 Trace
  │                              input = {"query": "红烧肉怎么做", "top_k": 5}
  │
  ├─ rewriter.classify(query)
  │    └─ LLM 调用（带 callback: LangchainCallbackHandler）
  │         → 子 Span: ChatOpenAI  (tokens, latency)
  │
  ├─ _retrieve(intent_result, top_k, query)
  │    └─ FAISS + BM25 混合检索
  │         → contexts = [Document(...), Document(...), ...]
  │
  ├─ generator.generate(query, contexts, intent)
  │    └─ LLM 调用（带 callback: LangchainCallbackHandler）
  │         → 子 Span: ChatOpenAI  (tokens, latency)
  │
  ├─ _enrich_trace(query, intent_result, contexts, answer, elapsed)
  │    └─ lf.update_current_span(
  │         name="RAG: 红烧肉怎么做",
  │         metadata={
  │           "intent": "howto",
  │           "target_dish": "红烧肉",
  │           "num_chunks": 5,
  │           "contexts": [...],
  │           "tag_intent": "howto",
  │           ...
  │         }
  │       )
  │
  └─ return answer  →  Trace output = "红烧肉的做法：1. 五花肉切块..."
                        Trace 自动关闭，数据发送到 Langfuse
```

## 5. 依赖

```
langfuse>=2.0.0          # 已在 requirements.txt 中
```

## 6. 环境变量汇总

| 变量 | 用途 | 模块 |
|------|------|------|
| `LANGFUSE_PUBLIC_KEY` | Langfuse 项目公钥（必填） | tracing.py, pipeline.py |
| `LANGFUSE_SECRET_KEY` | Langfuse 项目密钥（必填） | tracing.py, pipeline.py |
| `LANGFUSE_BASE_URL` | Langfuse 服务地址（可选，默认 cloud.langfuse.com） | tracing.py（SDK 自动读取） |
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥（LLM 模式需要） | pipeline.py, llm_chain.py, llm_classifier.py |

## 7. 已知问题

### 7.1 langfuse v4.x API 差异

langfuse v4.7.0 使用 OpenTelemetry 作为底层追踪框架，API 与 v2/v3 有所不同：

- **没有 `update_current_trace` 方法**：应使用 `update_current_span()` 代替。`@observe` 创建的根 span 其 metadata 会自动关联到 trace
- **`update_current_span` 不支持 `tags`**：标签信息以 `tag_` 前缀的 metadata 字段存储，在 Dashboard 中可通过 metadata 搜索
- **`CallbackHandler` 需要 `langfuse>=2.0.0`**：当前 `requirements.txt` 中的版本约束确保兼容性

### 7.2 metadata 中 contexts 大小限制

检索到的 context 全文存储在 `metadata.contexts` 中。如果 context 很大（完整食谱文件可达数千字），可能超出 Langfuse metadata 的大小限制。当前实现存储完整文本，如遇到截断问题可改存前 N 字符的摘要。

## 8. 涉及的改动文件

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `src_langchain/tracing.py` | **新增** | Langfuse CallbackHandler 初始化模块 |
| `src_langchain/pipeline.py` | 修改 | `@observe` 装饰器、`_enrich_trace()` metadata 注入、callback 传递 |
| `src_langchain/config.py` | 修改 | 新增 `LANGFUSE_PUBLIC_KEY_ENV` 等环境变量配置 |
| `src_langchain/generation/llm_chain.py` | 修改 | 新增 `callbacks` 参数，传入 LLM 调用 config |
| `src_langchain/rewriting/llm_classifier.py` | 修改 | 新增 `callbacks` 参数，传入 LLM 调用 config |
| `src_langchain/rewriting/rewriter.py` | 修改 | 工厂方法 `get_intent_classifier` 新增 `callbacks` 参数 |
| `requirements.txt` | 修改 | 新增 `langfuse>=2.0.0` |

## 9. 下一步可以做的事

- [ ] **用户反馈收集**：在 Gradio UI 添加 👍/👎 按钮，调用 `lf.create_score()` 记录用户打分
- [ ] **Datasets + Experiments**：将 `data/evaluation/test_queries.yaml` 导入 Langfuse Dataset，运行实验对比不同配置
- [ ] **Prompt 管理**：将 4 套 prompt 模板迁移到 Langfuse Prompt Management，支持热更新和版本回滚
- [ ] **Session 追踪**：在聊天模式中添加 `session_id`，关联多轮对话
- [ ] **成本仪表盘**：在 Langfuse UI 中配置 DeepSeek 模型定价，自动统计每查询/每意图的成本
- [ ] **LLM-as-Judge 自动评分**：使用 RAGAS 指标对 trace 进行质量评分，分数回传 Langfuse（`docs/langfuse-integration.md` 的早期版本曾包含此功能原型，已回退；正确实现需解决 `get_current_trace_id()` 在 `@observe` 上下文中返回 `None` 的问题）
