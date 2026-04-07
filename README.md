# AI-RAG 个人知识助手（LangChain + Chroma + 代码 Skill）

独立的小型 RAG 演示项目：把文本/文件写入向量库，再基于检索结果回答；支持 **LLM 总结链（LCEL）** 与 **规划型代码 Skill（动态步骤 + 顺序执行）**。

## 功能概览

- **入库**：粘贴文本或上传 `.txt` / `.md`，切块后写入 **Chroma**。
- **对话**：`POST /chat` 返回 `answer`、`sources`、`used_skill`。
- **路由**（`rag_assistant.py`）：
  1. 先由 **LLM 路由链** 判断走 `agent` 还是 `skill`（JSON：`route` / `reason` / `plan_steps`）。
  2. 若用户问题命中「规划/计划/步骤/roadmap」等关键词，会 **强制 `route=skill`**。
  3. **`skill` 分支**：直接调用 `run_code_skill` → `skills/planner_skill.py`（见下文）。
  4. **`agent` 分支**：进入 `langchain.agents.AgentExecutor`，由 `create_tool_calling_agent` 自动决定调用 `search_knowledge` / `make_knowledge_card` / `run_code_skill`。

> 说明：当前实现已使用 **`langchain.agents.AgentExecutor`** 作为主执行器。

## 技术栈

- Flask（Web + 页面）
- LangChain（`AgentExecutor`、`create_tool_calling_agent`）
- LangChain Core（`ChatPromptTemplate`、LCEL、`@tool`）
- langchain-openai（OpenAI 兼容接口，如 DeepSeek）
- langchain-chroma、ChromaDB
- sentence-transformers / HuggingFace Embeddings（BGE 等）

## 目录与关键文件

| 路径 | 说明 |
|------|------|
| `app.py` | Flask 路由：`/`、`/health`、`/ingest/text`、`/ingest/file`、`/chat` |
| `rag_assistant.py` | RAG 核心：路由、检索、总结链、工具封装、`chat()` |
| `config.yaml` | LLM、向量模型、Chroma、分块与 `top_k` |
| `templates/index.html` | 简单 Web 页面：入库 + 对话 |
| `skills/planner_skill.py` | 代码 Skill：`@skill_step` 注册步骤、`SequentialPlanner`、**LLM 动态规划步骤**后顺序执行 |

## 配置说明（`config.yaml`）

- **`llm`**：`model`、`api_key`、`base_url`、`temperature`、`timeout_s`（OpenAI 兼容服务）。
- **`embeddings`**：`model`（如 `BAAI/bge-small-zh-v1.5`）、`cache_dir`（模型缓存目录）。
- **`vectorstore`**：`persist_dir`（Chroma 持久化）、`collection_name`。
- **`rag`**：`chunk_size`、`chunk_overlap`、`top_k`。

请自行将 `api_key` 等敏感信息填入本地配置，**不要将真实密钥提交到仓库**。

## 安装与运行

```bash
cd D:\otherwise\AI-RAG
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

浏览器打开：`http://localhost:8000`

## Web 页面布局

- **导入知识（文本）**
  - `source`：来源标识，写入文档 metadata，检索结果里会出现在 `sources`。
  - 大文本框：要入库的正文。
  - **导入文本**：调用 `POST /ingest/text`。
- **导入知识（文件）**
  - 仅支持 `.txt` / `.md`。
  - **上传并导入**：`POST /ingest/file`。
- **对话**
  - 上方为对话日志；下方输入问题后 **发送**：`POST /chat`。
  - 日志会显示 `sources` 与 `used_skill`（是否走了 Skill 分支）。

## HTTP 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 首页 |
| `GET` | `/health` | 健康检查 |
| `POST` | `/ingest/text` | JSON：`{ "text", "source" }` |
| `POST` | `/ingest/file` | `multipart/form-data`，字段 `file` |
| `POST` | `/chat` | JSON：`{ "message" }` → `{ "answer", "sources", "used_skill" }` |

## 代码 Skill（`planner_skill.py` ）

- 使用 **`@skill_step`** 注册可执行步骤（如 `collect_background`、`extract_key_points` 等）。
- **`plan_steps_dynamically`**：把 **可选步骤 id 列表** + 用户问题 + 检索摘要交给 LLM，输出 JSON `steps`；代码侧过滤为仅注册步骤，再交给 **`SequentialPlanner`** 顺序执行。
- 若 LLM 输出无法解析或步骤为空，会回退到默认步骤顺序。

Skill 工具输出末尾可带 `[SKILL_EXECUTED]`，便于区分是否命中代码 Skill。

## 依赖与版本提示

- 建议使用 **Python 3.11** 或 **3.12**；若使用 **Python 3.14+**，可能出现 LangChain / Pydantic 相关告警，属上游兼容问题。
- `requirements.txt` 中 `langchain` 及相关包建议保持同一主版本线，避免 API 变更导致导入失败。
- 若看到 `HuggingFaceEmbeddings was deprecated`，表示当前仍在使用 `langchain_community` 里的旧入口；功能可用，后续可迁移到 `langchain-huggingface`。
- 若看到 `Failed to send telemetry event ...`（来自 Chroma），通常不影响入库和检索，可先忽略。
