# RAGForge 架构与功能说明

本文档说明 RAGForge 的模块划分、各模块职责、关键数据流与设计决策，作为代码的补充说明。

---

## 1. 总体分层

系统分为四层，层与层之间通过接口抽象解耦，具体实现可替换：

```
┌─────────────────────────────────────────────────────┐
│ 接入层    api/         FastAPI 路由、中间件、限流     │
├─────────────────────────────────────────────────────┤
│ 编排层    query/ agent/ retrieval/ generation/       │
│           guardrails/ cache/                         │
├─────────────────────────────────────────────────────┤
│ 核心抽象  core/        LLM / 向量存储 / embedding 接口│
├─────────────────────────────────────────────────────┤
│ 实现层    providers/    OpenAI / BGE / Milvus / ES    │
│ 横切      config/ observability/ eval/ ingestion/     │
└─────────────────────────────────────────────────────┘
```

核心原则：**上层只依赖抽象，不直接 import 具体客户端**（如业务代码不 import milvus/elasticsearch/openai 客户端），便于换后端与单测隔离。

---

## 2. 模块职责

### 2.1 config —— 配置
- `Settings`（pydantic-settings）：`RAGFORGE_` 前缀环境变量 + `.env` 文件，`SecretStr` 保护 API key，`env_ignore_empty` 让空值回退默认。
- `get_settings()` 进程级单例。

### 2.2 ingestion —— 文档入库
- **parsers**：PDF(pymupdf) / Markdown / Word(python-docx) / HTML 四类解析器，统一输出 `ParsedDocument`（doc_id / title / sections / tables / metadata）。标题树由共享的 `HeadingStack` 维护，PDF 用字号启发式识别标题层级。
- **chunking**：三种策略
  - `StructureChunker`：按 section 切分，超预算按段落继续切；
  - `SemanticChunker`：相邻句子 embedding 余弦相似度低于阈值处切分；
  - `ParentChildChunker`：先切 parent(~800 tokens) 再切 child(~200 tokens)，child 通过 `parent_id` 关联 parent。
  - 共性：token 预算可注入计数器、overlap 可配、chunk_id = `sha1(doc_id + heading_path + 序号)` 幂等。

### 2.3 core —— 核心抽象
- **core/llm**：`Message` / `LLMResult` / `BaseLLM`（complete / stream / complete_structured）+ 错误分类学（超时/连接/5xx 为可重试）。`FallbackLLM` 多 provider 降级 + 指数退避重试 + 熔断（连续 5 次失败熔断 60s）。
- **core/embeddings**：`EmbeddingProvider`（embed / embed_query），批次切分、维度校验、doc 侧缓存。
- **core/vector_store**：`VectorStore`（add / search / search_text / search_hybrid / delete），`Filter`（doc_id + metadata 字段，权限隔离），RRF 融合。

### 2.4 providers —— 具体实现
- `OpenAILLM` / `OpenAIEmbedding`：OpenAI 兼容协议，base_url 可切 DeepSeek / 豆包。
- `BGEEmbedding` / `BGEReranker`：本地 sentence-transformers，query/passage 前缀，可选 local extra 安装。
- `MilvusVectorStore`：HNSW/COSINE 索引，JSON 字段过滤，upsert 幂等。
- `ElasticsearchStore`：BM25 全文 + dense kNN 双检索，RRF 混合。

### 2.5 retrieval —— 检索
- `DenseRetriever`（向量）/ `SparseRetriever`（BM25）/ `HybridRetriever`（RRF 融合，两路并发）。
- `Reranker` 抽象 + `BGEReranker`（Cross-Encoder 逐对打分）。
- `RetrievalPipeline`：召回 top-50 → 重排 → 截断 top-8 两级漏斗。
- `self_rag`：`SelfRagEvaluator` 评估检索质量（sufficient/retry/insufficient + 逐 chunk relevance），`CorrectiveRagRetriever` 据判定改写重试（≤2 轮）或明确拒答。

### 2.6 query —— 查询理解
- `IntentRouter`（意图分类）、`QueryRewriter`（多轮指代消解）、`QueryExpander`（多查询扩展）、`HydeGenerator`（假想文档）。
- `QueryUnderstandingService` 编排以上步骤，各步可配置启停，改写结果流入扩展与 HyDE。

### 2.7 generation —— 生成
- `ContextAssembler`：按分数降序 + 去重 + 父子块还原把上下文压进 token 预算，并分配 `[1][2]` 编号。
- `Generator`：渲染 prompt 调用 LLM，解析答案引用，返回 `GenerationResult`（answer / citations / token / 耗时 / 成本）。

### 2.8 cache —— 缓存
- `CacheService`：精确命中（md5 直查）+ 语义命中（embedding 余弦 ≥ 0.92 阈值扫描）。
- 文档更新按 doc_id 主动失效；敏感内容（邮箱/手机号/身份证）拒绝缓存。

### 2.9 guardrails —— 护栏
- `InputGuard`（safe/harmful/injection/out_of_scope）、`OutputGuard`（safe/hallucination/unsafe），LLM 语义判断，解析失败 fail-closed 拦截。

### 2.10 agent —— 多跳 Agent
- `Planner`（拆解子问题 + 指定工具）、`Executor`（检索 / 安全计算器 / 查表三工具）、`Reflector`（ok/revise）。
- `AgenticRagEngine`：plan → execute → reflect → 修正重试，最大步数/轮数硬上限防死循环。

### 2.11 observability —— 可观测
- OpenTelemetry span（`@traced` 装饰器自动开 span，记录 latency/error/语义属性）、Prometheus 指标（qps/延迟分位/错误率/缓存命中率/成本）、structlog 日志（trace_id/request_id 关联）。

### 2.12 eval —— 评估
- 检索指标纯函数（recall_at_k / precision_at_k / mrr / hit_rate）。
- 生成指标 LLM-as-Judge（faithfulness / answer_relevance）。
- 评测 CLI：`python -m ragforge.eval run`，输出 JSON 报告（总指标 + 难度分层 + 失败样例 + 基线 diff）。

### 2.13 api —— 接口
- `POST /v1/chat`（非流式问答）、`POST /v1/chat/stream`（SSE）、`POST /v1/documents`（Celery 异步入库）、`GET /v1/documents/{id}`、`GET /v1/health`、`GET /v1/metrics`。
- 统一异常处理、Pydantic 校验、X-Request-ID 透传、令牌桶限流。

---

## 3. 一次查询的生命周期

```
POST /v1/chat {query, history}
  1. 输入护栏：LLM 判断 safe/harmful/injection/out_of_scope，block 直接 400 拦截
  2. 缓存：精确命中或语义命中（相似度 ≥0.92）→ 直接返回，零 LLM 调用
  3. 查询理解：意图路由 + 改写（消解指代）+ 多查询扩展 + HyDE
  4. 检索：Dense + Sparse 并发召回 → RRF 融合 → Self-RAG 评估
     ├─ insufficient → 明确拒答"资料不足"
     └─ retry → 用改写后的查询重试（≤2 轮）
  5. 重排：Cross-Encoder 对 top-50 精排，取 top-8
  6. 生成：token 预算内装配上下文（[1][2] 编号）→ 生成带引用答案
  7. 输出护栏：LLM 判断幻觉/不安全，block 拦截
  8. 写缓存 + 记录 cost/trace
  9. 响应 {code, data:{answer, citations}, trace_id, cost}
```

整个过程每个阶段都有 span（`rag.rewrite` → `rag.retrieve.dense/sparse/hybrid` → `rag.self_rag` → `rag.rerank` → `rag.generate`），Jaeger 中可见完整调用树。

---

## 4. 核心数据结构

| 结构 | 字段 | 说明 |
| --- | --- | --- |
| `Message` | role, content | 统一消息，API 层禁止裸 dict |
| `Chunk` | chunk_id, doc_id, text, parent_id, heading_path, page, metadata | 检索/生成的单位 |
| `SearchHit` | chunk_id, score, chunk | 检索命中 |
| `Citation` | chunk_id, page, text, score, doc_id | 答案引用，可反查 chunk |
| `GuardResult` | verdict(pass/block), category, reason | 护栏判定 |

---

## 5. 工程化约定

- **依赖管理**：uv，`pyproject.toml` 单源，可选依赖 `local`（sentence-transformers）。
- **类型**：mypy strict（85 个源文件全绿）。
- **测试**：pytest + pytest-asyncio + pytest-cov，261 个用例，总覆盖率 95.84%，核心模块 ≥90%，CI 以 `--cov-fail-under=85` 门禁。
- **代码风格**：ruff（E/F/I/UP/B/SIM），pre-commit 钩子（ruff + mypy）。
- **提示词管理**：全部外置在 `data/prompts/`（12 个模板），不在代码里硬编码。
