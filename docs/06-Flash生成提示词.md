# 06 · Flash 生成提示词（可直接投喂）

> 本文件是把整个项目"翻译"成**一段段可直接复制给 Flash / Cursor 的生成指令**。
> 每段提示词都是**自包含**的：Flash 看不到本项目的其它文档，所以背景、接口、验收标准都要写全。
>
> **使用顺序**：按编号从上到下。建议**一次只投喂一段**，等代码生成 + 测试通过后，再投喂下一段（把上一段的产物路径告诉 Flash 以保持上下文）。

---

## 通用模板（每一段都要包含这些要素）

```
[角色] 你是一名资深 Python 后端工程师，精通 RAG 系统与工程化最佳实践。
[背景] （本模块在整个系统中的位置、依赖什么）
[任务] （具体要实现什么）
[技术约束] （Python 3.11、完整类型注解、ruff/mypy strict、不引入多余依赖）
[接口/数据结构] （明确的类名、方法签名、dataclass 字段）
[验收标准] （如何才算完成：能运行、能测试）
[测试要求] （要写哪些单测）
[不要做] （明确禁止：硬编码、裸 dict、跳过测试）
```

---

## F01 · 项目脚手架初始化

```text
[角色] 你是资深 Python 工程化专家。
[背景] 我要从零搭建一个工程化的 RAG 项目，包名 `ragforge`，用 `src/` 布局。
[任务] 初始化项目骨架：
1. 生成 `pyproject.toml`：Python >=3.11，依赖管理用 uv，声明依赖 fastapi、uvicorn、pydantic、pydantic-settings、openai、structlog、redis、celery、opentelemetry-api/sdk、pytest、pytest-asyncio、pytest-cov、ruff、mypy。
2. 配置工具链：`[tool.ruff]` line-length=100、select=["E","F","I","UP","B","SIM"]；`[tool.mypy]` strict=true。
3. 建立目录：src/ragforge/{config,core,providers,pipeline,retrieval,query,generation,ingestion,cache,eval,observability,agent,guardrails,api}，每个目录建空 `__init__.py`。
4. 建 tests/{unit,integration,e2e}。
5. 建 `.gitignore`（忽略 .env、.venv、__pycache__、*.pyc、.pytest_cache、.mypy_cache）。
6. 建 `.pre-commit-config.yaml`：ruff + mypy。
7. 建 `Makefile`：`make lint`、`make type`、`make test`、`make fmt`。
[验收标准] 空项目能 `uv sync`、`make lint`、`make type` 通过。
[不要做] 不要写任何业务逻辑，只搭骨架。
```

---

## F02 · 配置模块

```text
[角色] 你是资深 Python 后端工程师。
[背景] 这是 RAG 项目的配置模块，位于 `src/ragforge/config/`，用 pydantic-settings。
[任务] 实现 `Settings` 类（env_prefix="RAGFORGE_"，env_file=".env"），至少包含：
- llm_provider(str)、llm_model(str)、llm_api_key(SecretStr)、llm_fallback_chain(list[str])
- embedding_model(str)、embedding_dim(int)、reranker_model(str)
- retrieval_top_k(int=50)、rerank_top_n(int=8)、rrf_k(int=60)
- semantic_cache_enabled(bool=True)、semantic_cache_threshold(float=0.92)
- otel_endpoint(Optional[str])
提供 `get_settings()` 返回单例；提供 `.env.example` 模板文件（值留空）。
[验收标准] 能读取环境变量与 .env；SecretStr 打印时被脱敏。
[测试] 写单测：验证 env 覆盖默认值、SecretStr 脱敏。
[不要做] 不要把真实 key 写进代码或 .env.example。
```

---

## F03 · LLM Provider 抽象

```text
[角色] 你是资深 Python 后端工程师，精通 LLM 应用。
[背景] 位于 `src/ragforge/core/llm/`，定义统一 LLM 接口，屏蔽 provider 差异。
[任务] 实现：
1. `Message` dataclass：role、content。
2. `LLMResult` dataclass：text、prompt_tokens、completion_tokens、cost、latency_ms。
3. `BaseLLM(ABC)` 抽象：`async complete(messages, temperature=0.0, max_tokens=None) -> LLMResult`、`async stream(messages, **kw) -> AsyncIterator[str]`、`async complete_structured(messages, schema: dict) -> dict`。
4. 在 `src/ragforge/providers/` 实现 `OpenAILLM(BaseLLM)`（用 openai 库，OpenAI 兼容协议，base_url 可配，适配 DeepSeek/豆包）。
5. 实现 fallback：`FallbackLLM(BaseLLM)` 接收 provider 列表，按顺序尝试，捕获超时/连接/5xx 错误降级到下一个，全部失败抛 `RAGForgeError(code="E_LLM_DOWN")`。
6. 指数退避重试（仅幂等错误），连续失败 5 次熔断 60s。
[验收标准] complete/stream/complete_structured 可运行；fallback 在首个 provider 失败时自动切到第二个。
[测试] 用 FakeLLM 测 fallback、熔断、结构化输出修复逻辑，禁止真调 API。
[不要做] 不要硬编码 API key；不要用裸 dict 传消息。
```

---

## F04 · Embedding Provider

```text
[角色] 你是资深 Python 后端工程师。
[背景] 位于 `src/ragforge/core/embeddings/`。
[任务] 实现：
1. `EmbeddingProvider(ABC)`：`embed(texts: list[str], batch_size=32) -> list[list[float]]`、`embed_query(text) -> list[float]`。
2. 在 `providers/` 实现 `OpenAIEmbedding`（兼容协议）与 `BGEEmbedding`（本地 SentenceTransformer，query 前缀 "query:"、doc 前缀 "passage:"）。
3. 维度校验：入库与查询维度不一致时抛 `RAGForgeError`。
4. doc embedding 缓存到本地/Redis（可选，用文本 hash 作 key）。
[验收标准] 批量向量化返回正确维度 list。
[测试] 用假模型测维度校验、批量切分、query/doc 前缀注入。
[不要做] 不要在主流程里写死某个具体模型。
```

---

## F05 · 文档解析

```text
[角色] 你是资深文档处理工程师。
[背景] 位于 `src/ragforge/ingestion/parsers/`，负责把 PDF/Word/Markdown/网页解析成结构化文本。
[任务] 实现：
1. `ParsedDocument` dataclass：doc_id、title、sections(list[Section])、tables(list[Table])、metadata(dict)。
2. `Section` dataclass：heading_path(list[str])、text、page。
3. `Parser(ABC).parse(path) -> ParsedDocument`。
4. 具体实现：`PDFParser`（用 pymupdf 或 mineru 接口，提取文本+页码）、`MarkdownParser`（解析标题树 `#/##/###` 为 heading_path）、`WordParser`（python-docx，解析标题层级）、`HtmlParser`（正文抽取，去导航脚本）。
5. 解析器注册表：按文件扩展名路由到对应 Parser。
[验收标准] 同一样例 PDF/MD/Word 均能产出带标题层级的 sections。
[测试] 用样例文件断言 heading_path 与 text 正确；未知扩展名抛错。
[不要做] 不要把解析逻辑写死在单个大函数里。
```

---

## F06 · 分块策略

```text
[角色] 你是资深 RAG 工程师，精通分块策略。
[背景] 位于 `src/ragforge/ingestion/chunking/`，这是 RAG 质量的关键。
[任务] 实现：
1. `Chunk` dataclass：chunk_id、doc_id、text、parent_id(Optional)、heading_path(list[str])、page(Optional)、metadata(dict)。
2. `Chunker(ABC).split(doc: ParsedDocument) -> list[Chunk]`。
3. 三种实现：
   - `StructureChunker`：按 section 切，保留 heading_path，超出 max_tokens 时按段落继续切。
   - `SemanticChunker`：基于相邻句子 embedding 余弦相似度找语义边界切分。
   - `ParentChildChunker`：先切 parent(约800 tokens)，再在 parent 内切 child(约200 tokens)，child 记录 parent_id。
4. 重叠窗口 overlap 参数可配。
5. chunk_id = sha1(doc_id + heading_path + 序号) 保证幂等。
[验收标准] 三种策略可跑通，child 的 parent_id 能关联到 parent。
[测试] 断言：chunk 不超长、heading_path 正确、parent-child 关联正确、幂等（同输入两次结果相同）。
[不要做] 不要固定写死分块大小，用配置。
```

---

## F07 · 向量库与文档库抽象

```text
[角色] 你是资深向量检索工程师。
[背景] 位于 `src/ragforge/core/vector_store/`，抽象向量库，屏蔽 Milvus/Qdrant 差异。
[任务] 实现：
1. `SearchHit` dataclass：chunk_id、score、chunk(Optional[Chunk])。
2. `VectorStore(ABC)`：`add(chunks) -> None`、`search(embedding, top_k, filters=None) -> list[SearchHit]`、`search_hybrid(embedding, text, top_k, filters=None)`、`delete(doc_id)`。
3. `providers/` 下实现 `MilvusVectorStore`（pymilvus，HNSW 索引，标量字段 doc_id）与 `ElasticsearchStore`（BM25 全文检索）。
4. `filters` 支持按 doc_id、metadata 字段过滤（文档权限隔离）。
5. 写入用 upsert 幂等。
[验收标准] 能 add 后 search 出结果；filters 能过滤。
[测试] 集成测试（需 docker 起 Milvus/ES，用 testcontainers 或 fixture）。
[不要做] 不要在上层直接 import milvus/elasticsearch 客户端。
```

---

## F08 · 查询理解

```text
[角色] 你是资深 RAG 工程师。
[背景] 位于 `src/ragforge/query/`，把用户原始问题转化为更适合检索的形式。
[任务] 实现 `QueryUnderstanding` dataclass（raw_query、intent、rewritten_query、expanded_queries、hyde_doc）与以下服务（都通过 llm 调用，prompt 见 data/prompts/）：
1. `IntentRouter.classify(query, history) -> intent`
2. `QueryRewriter.rewrite(query, history) -> str`（多轮指代消解）
3. `QueryExpander.expand(query, num=3) -> list[str]`（多查询）
4. `HydeGenerator.generate(query) -> str`
5. `QueryUnderstandingService.understand(query, history) -> QueryUnderstanding` 串联以上，每步可配置启停。
[验收标准] 输入模糊问题，输出完整 QueryUnderstanding；某步关闭时字段为 None。
[测试] 用 FakeLLM 返回固定 JSON，断言解析与串联逻辑；非法 JSON 时优雅降级。
[不要做] 不要把 prompt 硬编码在代码里，从 data/prompts 读取。
```

---

## F09 · 检索（多路召回 + RRF + 重排）

```text
[角色] 你是资深检索工程师。
[背景] 位于 `src/ragforge/retrieval/`。
[任务] 实现：
1. `DenseRetriever`、`SparseRetriever`、`HybridRetriever`，统一 `Retriever.retrieve(query, top_k, filters) -> list[SearchHit]`。
2. RRF 融合：`rrf_fuse(hit_lists, k=60) -> list[SearchHit]`，公式 score(d)=Σ 1/(k+rank_i(d))，去重取最高分。
3. `Reranker` 抽象 + `BGEReranker`（Cross-Encoder 对 query/chunk 打分精排）。
4. 两级漏斗：召回 top_k(50) → RRF → 重排 top_n(8)。
[验收标准] hybrid 检索能融合 dense+sparse 结果并重排。
[测试] 单测 rrf_fuse：两路都靠前的 chunk 融合后排第一；去重正确；重排用假 reranker 断言顺序。
[不要做] 不要只实现单一向量检索。
```

---

## F10 · Self-RAG / CRAG

```text
[角色] 你是资深 RAG 工程师。
[背景] 位于 `src/ragforge/retrieval/self_rag/`，生成前评估检索质量。
[任务] 实现：
1. `SelfRagAssessment` dataclass：verdict(sufficient/retry/insufficient)、relevance(list[bool])、refined_query(Optional[str])。
2. `SelfRagEvaluator.evaluate(query, chunks) -> SelfRagAssessment`（llm 批量判断，prompt P6）。
3. `CorrectiveRagRetriever`：verdict=retry 时用 refined_query 重新检索；insufficient 时返回空并标记"资料不足"。
[验收标准] 检索不足时自动改写重试，重试后仍不足则明确告知。
[测试] FakeLLM 返回三种 verdict，断言分支逻辑。
[不要做] 不要无脑重试超过 2 次（防死循环）。
```

---

## F11 · 生成（上下文装配 + 引用 + 流式）

```text
[角色] 你是资深 RAG 工程师。
[背景] 位于 `src/ragforge/generation/`。
[任务] 实现：
1. `Citation` dataclass：chunk_id、page、text、score。
2. `ContextAssembler.assemble(query, chunks, max_tokens) -> (context_str, citations)`：
   - 按分数降序填充直到 token 预算用尽；去重（相似 chunk 只留一条）；父子块还原（命中 child 取 parent）；给每块分配 [1][2] 编号。
3. `Generator.generate(query, chunks) -> GenerationResult`：装配上下文 → 渲染 prompt(P7) → 调用 llm（流式可选）→ 解析引用 → 返回 answer + citations + token/耗时/成本。
4. `GenerationResult` dataclass：answer、citations、prompt_tokens、completion_tokens、latency_ms、cost。
[验收标准] 答案带 [1][2] 引用，citations 能反查到 chunk。
[测试] FakeLLM 返回含引用的固定答案，断言 token 预算截断、去重、编号映射。
[不要做] 不要超过 max_context_tokens 预算。
```

---

## F12 · 语义缓存

```text
[角色] 你是资深后端工程师。
[背景] 位于 `src/ragforge/cache/`，用 Redis 做精确 + 语义缓存降本提速。
[任务] 实现：
1. `CacheService`：`get(query) -> Optional[CachedAnswer]`、`set(query, answer, citations)`。
2. 精确缓存：key = md5(query)。
3. 语义缓存：把 query embedding 与 Redis 中已存 query 做相似度匹配，超过 threshold(0.92) 命中。
4. `CachedAnswer` dataclass：answer、citations、source_query、hit_type(exact/semantic)。
5. 文档更新时按 doc_id 主动失效。
[验收标准] 相同/相似问题二次查询命中缓存，不调 LLM。
[测试] 用 fakeredis 测命中/未命中、阈值边界、失效逻辑。
[不要做] 不要缓存带用户隐私的内容（缓存前做脱敏标记）。
```

---

## F13 · 评估模块

```text
[角色] 你是资深 LLM 评测工程师。
[背景] 位于 `src/ragforge/eval/`，详见 docs/05-评估体系.md。
[任务] 实现：
1. 检索指标：`recall_at_k`、`precision_at_k`、`mrr`、`hit_rate`（纯函数）。
2. 生成指标：集成 RAGAS（faithfulness、answer_relevance）或自研 LLM-as-judge（prompt P10/P11）。
3. CLI：`python -m ragforge.eval run --dataset data/golden/qa.jsonl --retriever hybrid --metrics ...`，输出 JSON 报告（总指标 + 按难度分层 + 失败样例）。
4. 支持与上次报告 diff 对比。
[验收标准] 能跑完整个评测并输出报告 JSON。
[测试] 用构造的假 hits 断言 recall/mrr 计算正确。
[不要做] 不要硬编码评测集路径。
```

---

## F14 · 可观测性

```text
[角色] 你是资深可观测性工程师。
[背景] 位于 `src/ragforge/observability/`，三件套 trace/metric/log。
[任务] 实现：
1. OpenTelemetry 初始化：把 query 全链路拆成 span（rag.rewrite/rag.retrieve.dense/rag.retrieve.sparse/rag.rerank/rag.self_rag/rag.generate），每个 span 记 query、top_k、latency_ms、scores、tokens。
2. Prometheus 指标：qps、延迟分位(P50/95/99)、错误率、缓存命中率、单查询成本。
3. structlog 结构化日志，统一带 trace_id/request_id。
4. 提供装饰器 `@traced(name)` 自动开 span。
[验收标准] 一次 query 在 Jaeger 能看到完整 span 树。
[测试] 断言装饰器正确传播 trace_id。
[不要做] 不要把观测逻辑散落在业务代码各处。
```

---

## F15 · 护栏

```text
[角色] 你是资深 LLM 安全工程师。
[背景] 位于 `src/ragforge/guardrails/`。
[任务] 实现：
1. `InputGuard.check(user_input) -> GuardResult`（prompt P12，分类 safe/harmful/injection/out_of_scope）。
2. `OutputGuard.check(context, answer) -> GuardResult`（prompt P13，幻觉 + 安全检测）。
3. `GuardResult` dataclass：verdict(pass/block)、category、reason。
4. 命中 block 时抛出 `RAGForgeError(code="E_GUARD_BLOCKED")`，API 层返回友好提示。
[验收标准] 注入类输入被拦截；无法被上下文支撑的答案被标记。
[测试] FakeLLM 返回各类 verdict，断言拦截分支。
[不要做] 不要把护栏做成纯关键词匹配（要有 LLM 语义判断）。
```

---

## F16 · FastAPI 接口

```text
[角色] 你是资深 FastAPI 工程师。
[背景] 位于 `src/ragforge/api/`。
[任务] 实现：
1. `POST /v1/chat`：非流式问答，请求 {query, history}，响应 {code, data:{answer,citations}, trace_id, cost}。
2. `POST /v1/chat/stream`：SSE 逐 token 推送。
3. `POST /v1/documents`：上传文档，投递 Celery 异步入库，返回 doc_id。
4. `GET /v1/documents/{id}`：查询入库状态。
5. `GET /v1/health`：健康检查（含下游依赖探活）。
6. `GET /v1/metrics`：Prometheus 指标。
7. 统一异常处理器、Pydantic 请求校验、X-Request-ID 透传、令牌桶限流。
[验收标准] 本地起服务能 curl 通 /v1/health 和 /v1/chat。
[测试] 用 httpx/TestClient 测各路由 + 异常响应。
[不要做] 不要写同步阻塞的接口（用 async）。
```

---

## F17 · Agentic RAG

```text
[角色] 你是资深 Agent 工程师。
[背景] 位于 `src/ragforge/agent/`，处理多跳复杂问题。
[任务] 实现：
1. `Planner.plan(query) -> list[Step]`（prompt P14，拆解子问题 + 指定工具）。
2. `Executor` 按步骤执行：检索工具、计算器工具、查表工具。
3. `Reflector.reflect(query, context, answer) -> verdict`（prompt P15，ok/revise）。
4. `AgenticRagEngine.run(query) -> answer`：plan → execute → reflect → 不足则修正重试（最多 2 轮）。
5. 可用 LangGraph 或自研状态机实现。
[验收标准] 一个需多跳推理的问题能拆解执行并给出答案。
[测试] FakeLLM 返回固定 plan，断言执行顺序与反思分支。
[不要做] 不要无限循环，设最大步数。
```

---

## F18 · 补测试与覆盖率

```text
[角色] 你是资深测试工程师。
[背景] 项目已实现大部分模块，现在要补测试。
[任务] 为每个模块补齐单元测试，目标是总覆盖率 >=85%，核心模块（retrieval/generation/chunking/cache）>=90%。
用 FakeLLM/FakeVectorStore 隔离外部依赖；为 RRF、token 预算、语义缓存、分块幂等写针对性测试；把评测集典型样例固化为 e2e 断言。
[验收标准] `pytest --cov` 输出覆盖率达标，CI 通过。
[不要做] 不要为了覆盖率写无断言的假测试。
```

---

## 投喂技巧

1. **顺序投喂**：F01 → F02 → ... → F18，别跳步。
2. **附上下文**：投喂 F09 时，把 F07 生成的 `vector_store` 文件路径告诉 Flash："在已有的 `src/ragforge/core/vector_store/base.py` 基础上实现..."。
3. **一次一个模块**：模块过大时让 Flash 先列方案，再实现。
4. **强制验收**：每段提示词都有"验收标准"和"测试要求"，让 Flash **先写测试再实现**（TDD），或至少生成后跑 `make test`。
5. **纠错循环**：生成后把报错原样贴回去，附上"按 docs/04-工程化规范.md 修改"。
