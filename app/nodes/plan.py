"""节点 4：Planner —— 把「给我做个菜谱」拆成可执行的工具调用序列。

首轮做完整规划（查指南 → 算营养 → 生成菜谱）；
反思不通过时只规划「修正任务」，带上评审意见，避免重复做无用功。
"""

from typing import Any

from app.config import settings
from app.state import DietState


def plan_node(state: DietState) -> dict[str, Any]:
    revision = state.get("revision_count", 0)
    profile = state.get("profile", {})
    constraints: list[str] = profile.get("constraints", []) or []
    ingredients: list[str] = state.get("ingredients", []) or []
    reflection = state.get("reflection", {}) or {}
    hint = reflection.get("hint", "")

    plan: list[dict[str, Any]] = []

    if revision == 0:
        query = f"{'、'.join(constraints) or '均衡膳食'} 的食材搭配与烹饪建议"
        plan.append(
            {
                "id": f"r{revision}-search",
                "tool": "search_diet_guide",
                "args": {"query": query, "top_k": 3},
                "status": "pending",
            }
        )

    plan.append(
        {
            "id": f"r{revision}-nutrition",
            "tool": "calculate_nutrition",
            "args": {"ingredients": ingredients},
            "status": "pending",
        }
    )
    plan.append(
        {
            "id": f"r{revision}-recipe",
            "tool": "generate_recipe",
            "args": {
                "ingredients": ingredients,
                "constraints": constraints,
                "revision": revision,
                "hint": hint,
            },
            "status": "pending",
        }
    )

    return {"plan": plan[: settings.max_subtasks], "step_index": 0}
