from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from skills.planner_skill import build_action_plan_skill


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class RagAssistant:
    def __init__(self, config_path: str = "config.yaml") -> None:
        self.cfg = load_config(config_path)

        emb_cfg = self.cfg.get("embeddings", {})
        llm_cfg = self.cfg.get("llm", {})
        vs_cfg = self.cfg.get("vectorstore", {})
        rag_cfg = self.cfg.get("rag", {})

        cache_dir = Path(emb_cfg.get("cache_dir", "models"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=emb_cfg.get("model", "BAAI/bge-small-zh-v1.5"),
            cache_folder=str(cache_dir),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        persist_dir = Path(vs_cfg.get("persist_dir", "data/chroma"))
        persist_dir.mkdir(parents=True, exist_ok=True)
        collection_name = vs_cfg.get("collection_name", "rag_knowledge")
        try:
            self.vectorstore = Chroma(
                collection_name=collection_name,
                persist_directory=str(persist_dir),
                embedding_function=self.embeddings,
            )
        except Exception as e:
            # 常见于 Chroma 升级后读取旧持久化目录失败（如缺少 _type 字段）
            if "_type" not in str(e):
                raise
            backup_dir = persist_dir.with_name(f"{persist_dir.name}_backup")
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.move(str(persist_dir), str(backup_dir))
            persist_dir.mkdir(parents=True, exist_ok=True)
            self.vectorstore = Chroma(
                collection_name=collection_name,
                persist_directory=str(persist_dir),
                embedding_function=self.embeddings,
            )

        self.top_k = int(rag_cfg.get("top_k", 4))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(rag_cfg.get("chunk_size", 500)),
            chunk_overlap=int(rag_cfg.get("chunk_overlap", 80)),
        )

        self.llm = ChatOpenAI(
            model=llm_cfg.get("model", "deepseek-chat"),
            api_key=llm_cfg.get("api_key"),
            base_url=llm_cfg.get("base_url"),
            temperature=float(llm_cfg.get("temperature", 0.2)),
            timeout=int(llm_cfg.get("timeout_s", 60)),
        )
        self.summary_chain = self._build_summary_chain()
        self.skill_planner_chain = self._build_skill_planner_chain()
        self.tools = self._build_tools()
        self.tool_map = {t.name: t for t in self.tools}
        self.agent_executor = self._build_agent_executor()

    def ingest_text(self, text: str, source: str = "manual_input") -> int:
        text = (text or "").strip()
        if not text:
            return 0
        parts = self.text_splitter.split_text(text)
        now = datetime.now(timezone.utc).isoformat()
        docs = [
            Document(
                page_content=chunk,
                metadata={"source": source, "created_at": now},
            )
            for chunk in parts
            if chunk.strip()
        ]
        if docs:
            self.vectorstore.add_documents(docs)
        return len(docs)

    def ingest_file(self, filename: str, file_bytes: bytes) -> int:
        if not file_bytes:
            return 0
        text = file_bytes.decode("utf-8", errors="ignore")
        source = f"upload:{filename}"
        return self.ingest_text(text=text, source=source)

    def _should_use_skill(self, query: str) -> bool:
        keys = ["规划", "计划", "学习路径", "roadmap", "plan", "步骤", "拆解"]
        q = (query or "").lower()
        return any(k in q for k in keys)

    def _build_summary_chain(self):
        summary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个知识总结助手。请把上下文整理成简洁、可执行的中文要点，"
                    "优先输出3-6条项目符号，避免编造。",
                ),
                (
                    "human",
                    "问题：{question}\n\n上下文：\n{context}\n\n请输出结构化总结。",
                ),
            ]
        )
        return summary_prompt | self.llm | StrOutputParser()

    def _retrieve(self, query: str) -> tuple[list[str], list[str]]:
        docs = self.vectorstore.similarity_search(query, k=self.top_k)
        chunks = [d.page_content for d in docs if d.page_content.strip()]
        sources = [str(d.metadata.get("source", "unknown")) for d in docs]
        return chunks, sources

    def _summarize_chunks(self, question: str, chunks: list[str]) -> str:
        if not chunks:
            return "未检索到可总结的知识片段。"
        context = "\n\n".join(chunks)
        text = self.summary_chain.invoke({"question": question, "context": context}).strip()
        return text
    

    def _build_skill_planner_chain(self):
        planner_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个路由规划器。你的任务是决定当前问题应该走普通agent流程，"
                    "还是走代码skill流程。只输出JSON，不要输出其他文本。"
                    '格式: {{"route":"agent|skill","reason":"...","plan_steps":["..."]}}',
                ),
                (
                    "human",
                    "用户问题：{question}\n"
                    "若问题是任务拆解、学习计划、执行步骤、路线图，route=skill；"
                    "其他问题 route=agent。",
                ),
            ]
        )
        return planner_prompt | self.llm | StrOutputParser()

    def _plan_route(self, question: str) -> dict[str, Any]:
        raw = self.skill_planner_chain.invoke({"question": question}).strip()
        default = {"route": "agent", "reason": "default", "plan_steps": []}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return default
        route = str(data.get("route", "agent")).strip().lower()
        if route not in {"agent", "skill"}:
            route = "agent"
        reason = str(data.get("reason", "")).strip() or "unspecified"
        plan_steps = data.get("plan_steps", [])
        if not isinstance(plan_steps, list):
            plan_steps = []
        return {"route": route, "reason": reason, "plan_steps": [str(s) for s in plan_steps]}

    def _build_tools(self):
        @tool
        def search_knowledge(query: str) -> str:
            """检索知识库并返回上下文片段与来源。"""
            chunks, sources = self._retrieve(query)
            return json.dumps(
                {"chunks": chunks, "sources": sources},
                ensure_ascii=False,
            )

        @tool
        def make_knowledge_card(query: str) -> str:
            """针对问题自动检索并输出结构化总结卡片。"""
            chunks, sources = self._retrieve(query)
            summary = self._summarize_chunks(query, chunks)
            source_lines = "\n".join(f"- {s}" for s in sources[:5]) or "- unknown"
            return f"【知识卡片】\n问题：{query}\n\n{summary}\n\n来源：\n{source_lines}"

        @tool
        def run_code_skill(query: str) -> str:
            """运行代码skill，输出结构化执行计划（planner触发）。"""
            chunks, sources = self._retrieve(query)
            plan_text = build_action_plan_skill(query, chunks, llm=self.llm)
            source_lines = "\n".join(f"- {s}" for s in sources[:5]) or "- unknown"
            return f"{plan_text}\n\n来源：\n{source_lines}\n\n[SKILL_EXECUTED]"

        return [search_knowledge, make_knowledge_card, run_code_skill]

    def _build_agent_executor(self) -> AgentExecutor:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个RAG Agent。你可以调用工具完成检索问答与总结。"
                    "事实问答优先调用 search_knowledge；总结类问题优先调用 make_knowledge_card；"
                    "仅在任务拆解/计划场景调用 run_code_skill。"
                    "上下文不足时必须明确不确定，并给出下一步建议。",
                ),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )

    @staticmethod
    def _extract_sources(intermediate_steps: list[Any]) -> list[str]:
        found: list[str] = []
        for _, observation in intermediate_steps:
            if not isinstance(observation, str):
                continue
            try:
                payload = json.loads(observation)
            except json.JSONDecodeError:
                continue
            srcs = payload.get("sources", [])
            if isinstance(srcs, list):
                for s in srcs:
                    txt = str(s).strip()
                    if txt and txt not in found:
                        found.append(txt)
        return found

    @staticmethod
    def _used_skill(intermediate_steps: list[Any]) -> bool:
        for action, _ in intermediate_steps:
            if getattr(action, "tool", "") == "run_code_skill":
                return True
        return False

    def chat(self, message: str) -> dict[str, Any]:
        question = (message or "").strip()
        if not question:
            return {"answer": "请输入问题。", "sources": [], "used_skill": False}
        plan = self._plan_route(question)
        if self._should_use_skill(question):
            plan["route"] = "skill"

        if plan["route"] == "skill":
            tool_out = self.tool_map["run_code_skill"].invoke({"query": question})
            answer = str(tool_out).strip()
            chunks, sources = self._retrieve(question)
            if not sources and chunks:
                sources = ["unknown"]
            return {"answer": answer, "sources": sources, "used_skill": True}
        out = self.agent_executor.invoke({"input": question})
        answer = str(out.get("output", "")).strip() or "暂时无法生成回答。"
        steps = out.get("intermediate_steps", [])
        sources = self._extract_sources(steps)
        used_skill = self._used_skill(steps)
        return {"answer": answer, "sources": sources, "used_skill": used_skill}
