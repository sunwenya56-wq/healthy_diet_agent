"""节点 5：Executor —— 按计划顺序调用工具，并把结果回填进共享状态。"""

from typing import Any

from app.state import DietState
from app.tools.registry import ToolRegistry

# 工具名 → (State 字段, 结果提取函数)。回填规则集中在一处，新增工具只需加一行
_RESULT_MAP = {
    "search_diet_guide": ("retrieved_knowledge", lambda r: r.get("hits", [])),
    "calculate_nutrition": ("nutrition_report", lambda r: r),
    "generate_recipe": ("draft_recipe", lambda r: r),
}


def execute_node(state: DietState, registry: ToolRegistry) -> dict[str, Any]:
    plan = [dict(item) for item in state.get("plan", [])]
    updates: dict[str, Any] = {}
    traces: list[dict[str, Any]] = []
    errors: list[str] = []

    for index, item in enumerate(plan):
        if item.get("status") != "pending":
            continue
        out = registry.call(item["tool"], item.get("args", {}))
        traces.append(out)

        if out["ok"]:
            item["status"] = "done"
            item["result"] = out["result"]
            slot = _RESULT_MAP.get(item["tool"])
            if slot:
                field, extract = slot
                updates[field] = extract(out["result"])
        else:
            item["status"] = "failed"
            errors.append(f"{item['tool']}: {out['error']}")

        plan[index] = item

    if traces:
        updates["tool_trace"] = traces
    if errors:
        updates["errors"] = errors
    updates["plan"] = plan
    return updates
