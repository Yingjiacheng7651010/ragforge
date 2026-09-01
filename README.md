# RAGForge

一个从零实现的工程化 RAG（检索增强生成）平台，覆盖**文档入库 → 混合检索 → 查询理解 → 生成 → 评估**的完整链路。

做这个项目的出发点，是解决通用 RAG 系统普遍存在的四个问题：

- **检索精度低** —— 单一向量召回对实体、编号、精确匹配类查询效果差
- **答案幻觉高** —— 生成模型在上下文不足时依然"硬编"答案
- **无法量化优化** —— 改了一堆参数却不知道到底有没有变好
- **缺乏生产级工程能力** —— 没有可观测、没有降级、没有成本控制

## 核心能力

| 模块 | 能力 |
| --- | --- |
| 检索 | 向量 Dense + BM25 Sparse 多路召回、RRF 融合、Cross-Encoder 重排两级漏斗、Self-RAG 检索自评纠错 |
| 查询理解 | 意图路由、多轮指代消解改写、多查询扩展、HyDE 假想文档生成 |
| 抗幻觉 | 上下文 Token 预算（严格不超）、强制 `[1][2]` 引用溯源、输出护栏（LLM 语义判幻觉） |
| 降本 | 精确(md5) + 语义(embedding 余弦 0.92 阈值)双层缓存，命中即零 LLM 调用，文档更新按 doc_id 主动失效 |
| 可观测 | OpenTelemetry 全链路 span 树、Prometheus 指标、structlog 日志 trace_id 关联 |
| Agent | 自研 plan → execute → reflect 状态机，多跳复杂问题拆解执行 |
| 评估 | 检索指标(recall@k/mrr/hit_rate) + LLM-as-Judge 生成指标，评测 CLI 支持难度分层报告与基线 diff |

## 系统架构

```
用户请求
   │
   ▼
API 层 (FastAPI) ── X-Request-ID / 令牌桶限流 / 统一异常处理
   │
   ├── 输入护栏 (LLM 语义判断 safe/harmful/injection/out_of_scope)
   │
   ▼
缓存层 ── 精确命中 → 直接返回
   │      └─ 语义命中 → 直接返回
   ▼
查询理解 ── 意图路由 + 查询改写 + 多查询扩展 + HyDE
   │
   ▼
检索层 ── Dense + Sparse → RRF 融合 → Self-RAG 纠错 → 重排(recall 50 → rerank 8)
   │
   ▼
生成层 ── Token 预算装配上下文 → 强制引用生成
   │
   ├── 输出护栏 (幻觉 / 不安全检测)
   │
   ▼
响应 (答案 + citations + trace_id + cost)
```

存储与模型通过接口抽象隔离，支持多后端：

- **向量存储**：Milvus（HNSW/COSINE）与 Elasticsearch（BM25 + dense kNN）双实现
- **LLM**：OpenAI 兼容协议（可切 DeepSeek / 豆包等），`FallbackLLM` 多 provider 降级
- **Embedding**：OpenAI 兼容端点 与 本地 BGE（query/passage 前缀）双实现

## 设计考量

几个关键决策背后的考虑：

1. **为什么多路召回 + RRF，而不是只调向量相似度？** 向量检索对语义相似敏感，但实体名、编号、专有名词这类精确匹配场景弱；BM25 正好互补。而 dense 的余弦分数与 BM25 分数量纲完全不同、无法直接加权，RRF 只关心排名（`1/(k+rank)`），融合稳定且免调权。

2. **为什么做两级漏斗（召回 50 → 重排 8）？** Cross-Encoder 精度高但逐对推理开销大，直接对所有文档打分不现实。先用轻量双路召回粗排，再对 top-50 精排，兼顾精度与成本。

3. **为什么 Self-RAG？** 检索结果"有"不代表"够"。让模型先判断检索质量（sufficient/retry/insufficient），不足时改写查询重试、仍不足则明确拒答，比"硬着头皮生成"更抗幻觉。

4. **为什么 Token 预算 + 强制引用？** 生成前先按分数、去重、父子块还原把上下文压进预算内，并要求答案挂 `[1][2]` 引用、引用必须能反查到 chunk——让幻觉可被追溯。

5. **为什么语义缓存要设阈值而不是只要相似就命中？** 阈值 0.92 是命中率与误命中率的权衡；同时配合"文档更新按 doc_id 主动失效"与"敏感内容（邮箱/手机号/身份证）拒绝缓存"，避免缓存错误答案或泄露隐私。

6. **为什么每个模块都要可评估？** RAG 优化最大的坑是"凭感觉调参"。所以检索有 recall@k/mrr，生成有 faithfulness/answer_relevance（LLM-as-Judge），且评测结果支持难度分层与基线 diff 对比——每个改动都有数据支撑。

## 技术栈

- **语言/框架**：Python 3.11+ / FastAPI / Pydantic v2 / Celery
- **检索/存储**：Milvus、Elasticsearch、Redis
- **LLM/Embedding**：openai（兼容协议）、sentence-transformers（本地 BGE）
- **可观测**：OpenTelemetry、Prometheus、structlog
- **工程化**：uv 依赖管理、ruff + mypy(strict)、pytest、pre-commit、Makefile

## 快速开始

```bash
# 安装依赖（首次）
uv sync

# 跑测试（含覆盖率门禁 85%）
uv run pytest

# 代码检查
make lint && make type

# 起服务（需先配置 .env 的 RAGFORGE_LLM_API_KEY 等）
cp .env.example .env   # 填真实值
uv run uvicorn ragforge.api.app:create_app --factory --host 127.0.0.1 --port 8000

# 跑评测
uv run python -m ragforge.eval run --dataset data/golden/qa.jsonl --metrics recall@5,mrr,hit_rate
```

## 项目结构

```
src/ragforge/
├── api/            # FastAPI 路由（chat/SSE/documents/health/metrics）、中间件、限流
├── agent/          # 多跳 Agent（plan/execute/reflect）
├── cache/          # Redis 精确 + 语义双层缓存
├── config/         # pydantic-settings 配置
├── core/           # LLM 抽象、向量存储抽象、embedding 抽象、错误
├── eval/           # 检索/生成指标、LLM-as-Judge、评测 CLI
├── generation/     # 上下文装配 + 引用生成
├── guardrails/     # 输入/输出 LLM 护栏
├── ingestion/      # 文档解析（PDF/MD/Word/HTML）+ 分块（结构/语义/父子）
├── observability/  # OTel / Prometheus / structlog
├── providers/      # OpenAI / BGE / Milvus / Elasticsearch 实现
├── query/          # 意图路由 / 改写 / 扩展 / HyDE
└── retrieval/      # Dense/Sparse/Hybrid 检索、重排、Self-RAG
data/
├── prompts/        # 12 个外置 prompt 模板
└── golden/         # 评测黄金集
tests/              # unit / integration / e2e（261 个用例）
```

更完整的模块职责、接口与数据流说明见 [ARCHITECTURE.md](ARCHITECTURE.md)。
