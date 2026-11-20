# 健康膳食 Agent

**LangGraph 状态机驱动的多模态健康膳食 AI Agent** —— 支持文本 / 语音 / 图像输入，从食材识别 → 营养分析 → 菜谱生成闭环决策。

<div align="center">

| 能力 | 实现 |
|---|---|
| 意图识别 / 菜谱生成 | DeepSeek (deepseek-chat) |
| 体检报告解析 | OCR → DeepSeek VLM 转录兜底 |
| 冰箱食材识别 | YOLOv11 (best.pt) → DeepSeek VLM 零样本兜底 |
| 语音转写 | 硅基流动 XingChenGSR-V1.0 |
| 语音播报 | Windows SAPI 系统语音（离线、免费） |
| 向量检索 | Milvus（生产）↔ 内存混合检索（本地降级） |
| 关系存储 | MySQL / SQLite（可切换） |
| 工具协议 | FastMCP stdio —— Agent 内部和外部 MCP Client 共用 |

</div>

---

## 架构一览

```
START
  │
  ▼
intake ──(带图片/报告)──► collect ──► analyze ──► plan ──► execute ──► reflect
  │                        │           │                              │
  │                    OCR/YOLO   指标抽取               评分 ≥ 80 分  │
  └──(纯文本)──────────────┘                                           │ 通过
                                                                   finalize ──► END
                                                                       ▲
                                                                       └── 未达标带 hint 回到 plan，最多 MAX_REVISIONS 轮
```

核心设计决策：
- **反思回环**不只是 retry —— Reflector 输出可解释的 100 分制评分（食材利用率 / 步骤完整度 / 约束覆盖 / 营养说明 / 热量区间），带 `hint` 触发 Planner 重规划，最多 `MAX_REVISIONS` 轮
- **约束是算出来的** —— 血糖 ≥6.1 → 控糖；血压 ≥140/90 → 低盐；胆固醇 / LDL / 甘油三酯超标 → 低脂；BMI ≥24 → 控热量；过敏史 → 避敏。不写死在 Prompt 里
- **状态即事实** —— 所有中间结果进 `DietState`（dataclass + TypedDict 混合），节点无副作用，可回放、可单元测试、可做可观测性
- **外部依赖全降级** —— LLM / VLM 没 key → Mock；YOLO 权重缺失 → VLM 兜底 → 样例数据；Milvus 连不上 → 内存向量库。主流程不因任何外部服务挂掉

## 目录结构

```
healthy_diet_agent/
├── app/
│   ├── config.py              全部配置 + 降级默认值
│   ├── state.py                DietState 定义
│   ├── graph.py                StateGraph 组装 + run_agent / stream_agent
│   ├── nodes/                  intake / collect / analyze / plan / execute / reflect / finalize
│   ├── tools/                  ToolRegistry (MCP 风格) + 5 个工具实现
│   ├── adapters/               LLM / Vision (OCR + YOLO + VLM 兜底) / Vector (Milvus ↔ 内存)
│   ├── db/models.py            SQLAlchemy：User / HealthReport / Ingredient / Recipe / AgentRun
│   ├── streaming.py            SSE 逐节点输出适配
│   ├── api.py                  FastAPI + 语音 ASR + TTS
│   ├── mcp_server.py           FastMCP stdio Server
│   └── static/index.html       Web Demo
├── data/
│   ├── guidelines/             《中国居民膳食指南》+ 慢性病饮食指南
│   └── uploads/                体检报告 / 冰箱图 / 语音（运行时写入）
├── models/best.pt              YOLOv11 训练好的食材权重
├── Dockerfile / Dockerfile.multi / docker-compose.yml
├── requirements.txt
└── tests/test_graph.py         指标抽取 / 约束 / 评审 / 端到端带修订
```

## 快速开始

### 1. 本地开发

```bash
pip install -r requirements.txt

# 全 Mock 模式 —— 零 key、零模型下载，全流程可跑（约 2 min）
python -m app.demo

# 起 FastAPI（http://127.0.0.1:8000）
uvicorn app.api:app --reload

# 跑测试
python -m pytest tests -q    # 6 passed
```

### 2. 接真实能力

复制 `.env.example` 为 `.env`，按需取消注释：

```bash
# LLM
LLM_MOCK=0
LLM_API_KEY=sk-your-deepseek-key
LLM_BASE_URL=https://api.deepseek.com/v1

# 多模态（真实 YOLO + OCR；不装则自动走 VLM 兜底）
VISION_MOCK=0     # 装 ultralytics + pytesseract
YOLO_MODEL_PATH=./models/best.pt

# 语音（硅基流动 ASR）
SILICONFLOW_API_KEY=sk-your-siliconflow-key
ASR_MODEL=XingChenAGI/XingChenGSR-V1.0

# 存储（切生产）
VECTOR_BACKEND=milvus
MILVUS_URI=http://localhost:19530
MILVUS_HISTORY_COLLECTION=user_history     # 用户历史交互向量化记忆
DATABASE_URL=mysql+pymysql://root:pass@127.0.0.1:3306/diet_agent?charset=utf8mb4
```

### 3. Docker Compose 一键启动（推荐生产用）

```bash
docker compose up -d --build
# 起 5 个容器：app + mysql:8.0 + milvus:2.4 (etcd + minio)
# FastAPI: http://localhost:8000
# MySQL:   localhost:3306    (root / rootpass)
# Milvus:  localhost:19530   (Attu http://localhost:9091)
```

容器内环境变量已在 `docker-compose.yml` 里预配好，换 key 只需改 `.env`。

### 4. 作为 MCP Server 被其他客户端调用

```bash
python -m app.mcp_server
```

暴露 5 个工具（stdio 协议）：`parse_health_report` / `detect_ingredients` / `search_diet_guide` / `calculate_nutrition` / `generate_recipe`。Claude Desktop、Dify、或自研调度器都能发现并调用。

## API 速查

| Method | Path | 说明 |
|---|---|---|
| GET  | `/healthz` | 健康检查 |
| GET  | `/api/tools` | 工具 schema 列表 |
| POST | `/api/upload/report` | 上传体检报告 → 返回解析文本 + 来源 (pdf/ocr/vlm/mock) |
| POST | `/api/upload/fridge` | 上传冰箱图 → 返回食材清单 + 来源 (yolo/vlm/mock) |
| POST | `/api/asr` | 语音转写 → 返回文本 + 来源 |
| POST | `/api/tts` | 文本合成语音 → 返回 base64 wav + 来源 |
| POST | `/api/agent/run` | 非流式 Agent 运行 |
| POST | `/api/agent/stream` | SSE 逐节点输出（intake → collect → analyze → plan → execute → reflect → finalize） |
| GET  | `/api/recipes` | 当前用户历史菜谱 |

## 已验证

- `python -m app.demo`：完整端到端 → 反思评审 → 修订一轮 → 最终菜谱
- `pytest tests -q`：**6 passed**（指标抽取 / 血糖 & 血压 & BMI 约束 / 过敏约束 / 评审打分 / 带修订的整图端到端 × 2 种报告）
- `tjbg_2.png`（糖尿病前期 160cm/68kg/BMI26.6/血糖6.7）+ `bx_2.jpg`：source=vlm，约束控糖/低脂/控热量，菜谱正确适配
- FastAPI：`/healthz`、`/api/agent/run`（自动落库 MySQL + 写入 Milvus 用户历史）、`/api/recipes` 均 200
- ASR：`XingChenAGI/XingChenGSR-V1.0` 直接吃 webm，中文识别正确
- TTS：Windows SAPI + 长文本分段合成，749 字答案出 174s wav
- MCP Server：5 个工具 schema 与 Agent 内部共用同一 ToolRegistry

## 设计亮点

1. **全链路降级** —— 外部依赖故障不会让 Agent 挂掉。连不上 Milvus → 内存向量库；没 YOLO 权重 → VLM 零样本兜底；LLM Mock 模式返回确定性假数据便于测试。
2. **反思不只是 retry** —— Reflector 用结构化可解释打分（食材利用率 / 步骤完整度 / 约束覆盖 / 营养说明 / 热量区间），带 hint 触发 Planner 重规划，最多两轮修订。
3. **约束驱动生成** —— 所有饮食约束从体检报告实时计算，不写死在 Prompt 里。新指标只需改 `collect.py` 的正则和阈值。
4. **统一工具协议** —— MCP Server 和 Agent 内部共享 ToolRegistry，一次实现两种调用方式。Dify / Claude Desktop 可以直接编排这 5 个工具。
5. **用户记忆向量化** —— 每次 Agent 运行后，final_answer + query + intent 自动写入 Milvus `user_history` collection，后续可做 query 改写或个性化参考。

## 技术栈

- **Agent 编排**：LangGraph StateGraph + LangChain Core
- **LLM / VLM / ASR**：DeepSeek / 硅基流动 OpenAI 兼容接口
- **视觉**：Ultralytics YOLOv11 (YOLO) + Tesseract / DeepSeek VLM (OCR)
- **语音**：XingChenGSR-V1.0 (ASR) + Windows SAPI (TTS)
- **向量**：Milvus pymilvus（生产）↔ 字符 n-gram + 关键词 BM25 内存检索（本地）
- **存储**：SQLAlchemy + MySQL（生产）↔ SQLite（本地降级）
- **服务**：FastAPI + Uvicorn + SSE Streaming
- **部署**：Docker Compose（app + mysql:8.0 + milvus:2.4 + etcd:3.5 + minio）
- **协议**：FastMCP stdio

## License

MIT
