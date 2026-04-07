"""Planner-style Skill：装饰器注册 + LLM 动态规划 + 顺序执行。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


SkillFn = Callable[["SkillContext"], str]
_SKILL_REGISTRY: dict[str, SkillFn] = {}


@dataclass
class SkillContext:
    question: str
    chunks: list[str]
    artifacts: dict[str, str]


def skill_step(name: str) -> Callable[[SkillFn], SkillFn]:
    """核心装饰器：注册 skill 步骤。"""

    def decorator(func: SkillFn) -> SkillFn:
        _SKILL_REGISTRY[name] = func
        return func

    return decorator


class SequentialPlanner:
    """顺序规划器：按 plan 中步骤依次执行 skill。"""

    def __init__(self, steps: Sequence[str]) -> None:
        self.steps = list(steps)

    def run(self, ctx: SkillContext) -> list[tuple[str, str]]:
        outputs: list[tuple[str, str]] = []
        for step_name in self.steps:
            fn = _SKILL_REGISTRY.get(step_name)
            if fn is None:
                outputs.append((step_name, f"[跳过] 未注册步骤: {step_name}"))
                continue
            result = fn(ctx).strip()
            outputs.append((step_name, result))
            ctx.artifacts[step_name] = result
        return outputs


@skill_step("collect_background")
def collect_background(ctx: SkillContext) -> str:
    if not ctx.chunks:
        return "暂无检索资料，先补充至少2条基础文档。"
    return f"背景信息：{ctx.chunks[0][:120]}"


@skill_step("extract_key_points")
def extract_key_points(ctx: SkillContext) -> str:
    if len(ctx.chunks) <= 1:
        return "关键点不足，先列出输入、输出、边界条件。"
    items = "；".join(x[:80] for x in ctx.chunks[1:3])
    return f"关键点：{items}"


@skill_step("design_minimal_task")
def design_minimal_task(ctx: SkillContext) -> str:
    seed = ctx.chunks[-1][:100] if ctx.chunks else "从最小用例开始"
    return f"最小任务：基于“{seed}”实现端到端最小闭环。"


@skill_step("define_validation")
def define_validation(ctx: SkillContext) -> str:
    return "验证方案：定义1个成功标准、1个失败样例、1个下一步优化。"


def _default_static_steps() -> list[str]:
    return [
        "collect_background",
        "extract_key_points",
        "design_minimal_task",
        "define_validation",
    ]


def _chunk_preview(chunks: Sequence[str], max_chars: int = 800) -> str:
    parts = [" ".join(c.strip().split()) for c in chunks if c and c.strip()]
    text = "\n---\n".join(parts[:6])
    return text[:max_chars] if text else "（无检索片段）"


def _strip_json_fenced(raw: str) -> str:
    s = raw.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", s)
    if m:
        return m.group(1).strip()
    return s


def _parse_dynamic_plan(raw: str) -> tuple[list[str], str]:
    try:
        data = json.loads(_strip_json_fenced(raw))
    except json.JSONDecodeError:
        return [], "JSON解析失败，使用默认步骤序列。"
    steps = data.get("steps", [])
    if not isinstance(steps, list):
        return [], "steps 非数组，使用默认步骤序列。"
    reason = str(data.get("reason", "")).strip() or "（无说明）"
    out = [str(s).strip() for s in steps if str(s).strip()]
    return out, reason


def _filter_registered_steps(
    proposed: Sequence[str], registry: dict[str, SkillFn]
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in proposed:
        if name in registry and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _build_dynamic_planner_chain(llm: Any):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是 skill 步骤规划器。下面给出**唯一允许**的步骤 id 列表，"
                "你必须从中选出一个**有序子序列**作为执行计划（可跳过不需要的步骤）。"
                "只输出合法 JSON，不要 markdown 围栏，不要其它文字。\n"
                '格式: {"steps":["id1","id2",...],"reason":"一句话说明为何这样排序/省略"}',
            ),
            (
                "human",
                "可选步骤 id（逗号分隔）：{available_ids}\n\n"
                "各步骤含义：\n"
                "- collect_background：从检索上下文提炼背景\n"
                "- extract_key_points：抽取关键信息/约束\n"
                "- design_minimal_task：设计最小可执行闭环任务\n"
                "- define_validation：定义验收与复盘方式\n\n"
                "用户目标：{question}\n\n"
                "检索片段摘要：\n{chunk_preview}\n\n"
                "请输出 JSON。steps 中的 id 必须完全来自可选列表，禁止编造新 id。",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


def plan_steps_dynamically(
    question: str,
    chunks: Sequence[str],
    llm: Any,
) -> tuple[list[str], str]:
    """LLM 产出步骤序列，过滤为仅已注册步骤；失败则回退默认顺序。"""
    available_ids = ", ".join(sorted(_SKILL_REGISTRY.keys()))
    preview = _chunk_preview(chunks)
    chain = _build_dynamic_planner_chain(llm)
    raw = chain.invoke(
        {
            "question": question,
            "chunk_preview": preview,
            "available_ids": available_ids,
        }
    )
    if not isinstance(raw, str):
        raw = str(raw)
    proposed, reason = _parse_dynamic_plan(raw.strip())
    filtered = _filter_registered_steps(proposed, _SKILL_REGISTRY)
    if not filtered:
        return _default_static_steps(), f"{reason}（已回退默认步骤）"
    return filtered, reason


def build_action_plan_skill(
    question: str,
    chunks: Sequence[str],
    llm: Any | None = None,
) -> str:
    """对外入口：有 llm 时动态规划步骤，否则使用固定顺序。"""
    cleaned = [" ".join(c.strip().split()) for c in chunks if c and c.strip()]
    ctx = SkillContext(question=question, chunks=cleaned, artifacts={})

    if llm is not None:
        step_names, plan_reason = plan_steps_dynamically(question, cleaned, llm)
        header = f"【Skill执行计划·动态规划】\n规划说明：{plan_reason}\n目标：{question}"
    else:
        step_names = _default_static_steps()
        header = f"【Skill执行计划·静态顺序】\n目标：{question}"

    planner = SequentialPlanner(steps=step_names)
    outputs = planner.run(ctx)
    body = "\n".join(f"- {name}: {text}" for name, text in outputs)
    return f"{header}\n{body}"
