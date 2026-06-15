# 食谱 RAG 智能问答系统

**Python · FAISS · BM25 · RRF · BGE-Reranker · LangChain · DeepSeek · FastAPI · SSE · Gradio · RAGAS · Langfuse**

基于混合检索与 LLM 生成的中文食谱 RAG 问答系统，363+ 道菜谱知识库。

## 1. 分块与 Small-to-Big 上下文增强

自研层级 Markdown 分块器，利用食谱 `#` `##` `###` 标题结构拆分为 L1（菜品）/ L2（章节）/ L3（子步骤）三级粒度。检索时用细粒度 chunk 精确命中，生成时检测目标菜名后从源文件读取完整食谱注入上下文，让 LLM 同时看到原料+步骤全貌，消除分块片段缺失信息导致的幻觉。

## 2. 多阶段检索 — RRF + Cross-Encoder 精排

三阶段检索架构：FAISS 稠密检索（Bi-Encoder）+ BM25 稀疏检索 → RRF 融合（k=60）→ Cross-Encoder 精排（BGE-reranker-v2-minicpm-layerwise）。Bi-Encoder 双塔模型独立编码 query 和 chunk 后求余弦相似度，速度快但交互不充分；Cross-Encoder 将 (query, chunk) 拼接送入 Transformer 全注意力计算，精度高但速度慢。采用粗排→精排的两阶段策略：RRF 融合 30 个候选后交给 Cross-Encoder 精排取 top-k，兼顾效率与精度。针对不同意图差异化路由：howto/ingredient 开精排（需精确匹配菜名和操作），recommendation 保持 MMR 多样性重排（需品类多样性而非单点精度）。

## 3. 混合检索 — RRF 融合

FAISS 稠密检索（text2vec-base-chinese, 768d）+ BM25 稀疏检索（jieba 分词），经 RRF（k=60）融合。选 RRF 而非线性加权：FAISS 余弦分数 ∈[-1,1]、BM25 分数无上界，两路量纲不可比，RRF 仅依赖排名天然规避归一化问题。意图驱动路由：推荐走多探针搜索+MMR 多样性重排，做法/原料偏重对应章节。

## 4. LangChain 框架

双后端架构（`app.py --backend` 切换）：原生 SDK 端手写 FAISS index.search() / BM25Okapi / RRF / OpenAI SDK 深入底层机制；LangChain 端基于 FAISS.from_documents() + BM25Retriever + RRFFusionRetriever + ChatOpenAI + StrOutputParser 构建，对接 Langfuse 可观测性。双后端镜像结构，共享 jieba 分词和 RAGAS 评估模块。

## 5. SSE 流式输出与 FastAPI 服务化

DeepSeek API `stream=True` 逐 token 返回，Pipeline 层新增 `run_stream()` 生成器方法，yield 四阶段事件（rewrite / retrieve / generate / done）。FastAPI + sse-starlette 封装 `POST /chat` SSE 接口，Gradio Chat 组件接入流式，用户即刻看到逐字输出。对比常规同步生成 5-15s 的空白等待，流式显著改善交互体验。同时保留非流式 `run()` 方法作为规则模式兜底。

## 6. RAGAS 评估

5 指标评估流水线（context_precision / recall / faithfulness / relevancy / correctness），30 条中文测试查询含 ground truth 与知识库实际内容对齐。`adapt_prompts(language="chinese")` 翻译评估 prompt + pickle 缓存。评估器与后端解耦（构造器注入 pipeline），一套代码评估两种后端。

## 7. Langfuse 可观测性

`@observe` 捕获 pipeline 级 trace，`LangchainCallbackHandler` 经 callbacks 参数完整传递链（Pipeline → IntentClassifier → LLMGenerator），自动记录每次 LLM 调用的 token 用量、模型名、延迟。Trace 富化注入意图、目标菜名、chunk 数等业务元数据。

## 8. Gradio 展示

4 Tab Web UI（Pipeline 总览 / 检索演示 / 数据概览 / 技术架构）。懒加载 + 内存缓存解决 `demo.load` 扫描 363 文件耗时 8s 致 UI 卡死问题。

---

## 9. 面试可能考察点

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

**答**：① 支持更多 LLM provider（Anthropic, Ollama），当前仅 DeepSeek；② 流式输出（SSE/WebSocket），改善长回答等待体验；③ Query 扩展（HyDE / Query2Doc），提升稀疏检索的召回；④ 搜索结果重排序（Cross-Encoder Reranker），替代 RRF 提升精度；⑤ 知识库增量更新（CDC 监控食谱仓库变更）；⑥ Docker 化部署 + API 化（FastAPI 替换 Gradio）。

### Q9: 为什么 RRF 之后还要加 Cross-Encoder Reranker？

**答**：RRF 融合的是 Bi-Encoder (FAISS) 和 BM25 的排序结果，两者都不是精细的相关性判断——Bi-Encoder 独立编码 query 和 chunk，只有余弦相似度一个标量交互；BM25 是词袋匹配。Cross-Encoder 将 (query, chunk) 拼接后做全自注意力，每个 token 都能与另一句的每个 token 交互，相关度判断远更精确。但 Cross-Encoder 慢（每对都需完整前向），不能在全量索引上用，所以只在 RRF top-30 候选上精排。这和多阶段推荐系统"召回→粗排→精排"的思路一致。

### Q10: SSE vs WebSocket，为什么选用 SSE？

**答**：这个场景是单向推送（服务端→客户端推送 token），不需要客户端向服务端持续发送数据。SSE 基于 HTTP 协议，实现简单（标准 EventSource API），天然支持自动重连，不需要额外的心跳保活机制。WebSocket 适合双向实时通信（如聊天室），对于 LLM 流式输出过于复杂，还会增加运维复杂度（代理配置、连接管理）。

---

## 10. 项目规模

| 指标 | 数值 |
|------|------|
| 知识库规模 | 363+ 道中文家常菜 |
| 分块总数 | ~2000+ |
| 向量维度 | 768 |
| 意图类型 | 4 种（推荐/做法/原料/知识） |
| 评估样本 | 30 条（含 ground truth） |
| 评估指标 | 5 项 RAGAS 指标 |
| 代码量 | ~3000 行（双后端 + 共享模块） |
| 后端实现 | 2 套（原生 SDK / LangChain） |
| 开发周期 | 约 5 天 |
