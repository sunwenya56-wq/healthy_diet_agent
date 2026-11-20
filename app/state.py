"""Agent 全局状态定义。

LangGraph 的核心是「共享状态 + 节点读写」：节点只负责往 State 写增量，
流程控制交给图的边，而不是像老代码那样把 if/else 埋在路由里。
"""

import operator
from typing import Annotated, Any, Literal, TypedDict

Intent = Literal["generate_recipe", "nutrition_consult", "parse_report", "chitchat"]


def merge_dict(left: dict, right: dict) -> dict:
    """合并 dict 增量，避免节点互相覆盖。"""
    return {**left, **right}


class DietState(TypedDict, total=False):
    # ---- 输入 ----
    session_id: str
    user_id: int
    user_query: str
    report_image_path: str | None
    fridge_image_path: str | None

    # ---- 意图与画像 ----
    intent: Intent
    need_multimodal: bool
    profile: dict[str, Any]          # 身高体重/指标/饮食约束
    health_metrics: dict[str, Any]   # BMI / 血糖 / 血脂 / 血压 / 过敏

    # ---- 资源 ----
    report_text: str
    ingredients: list[str]
    retrieved_knowledge: list[dict[str, Any]]

    # ---- 计划与执行 ----
    plan: list[dict[str, Any]]       # [{id, tool, args, status, result}]
    step_index: int
    tool_trace: Annotated[list[dict[str, Any]], operator.add]

    # ---- 产出与反思 ----
    draft_recipe: dict[str, Any]
    nutrition_report: dict[str, Any]
    reflection: dict[str, Any]       # {score, passed, issues}
    revision_count: int
    final_answer: str

    # ---- 错误 ----
    errors: Annotated[list[str], operator.add]
