"""把 LangGraph 的 state update 流翻译成前端友好的 SSE 事件。

固定事件协议（不要在迭代中修改类型，否则前端解析会飘）：
    step_start   {step_id, title, revision}
    tool_call    {step_id, tool, args_summary}
    tool_result  {step_id, tool, ok, summary}
    step_end     {step_id, title, key}
    answer       {chunk}        # finalize 产出的 markdown 最终答案
    final        {result}       # 完整状态快照（落库用）
    done         {}

step_id 的规则是 `{node}_r{revision_count}`，同 step_id 的事件前端幂等更新，
修订轮 plan/reflect 会自动拿到新 id，UI 上新增卡片而不是覆盖。
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from app.graph import stream_agent as _raw_stream

_NODE_TITLES = {
    "intake": "🧠 理解意图",
    "collect": "📸 采集多模态信息",
    "analyze": "📊 抽取健康画像",
    "plan": "📋 规划第 {n} 轮",
    "execute": "🛠️ 执行工具调用（第 {n} 轮）",
    "reflect": "🔍 第 {n} 轮评审",
    "finalize": "✅ 整理最终输出",
}


def _title(node: str, revision: int) -> str:
    template = _NODE_TITLES.get(node, node)
    if "{n}" in template:
        return template.format(n=revision + 1)
    return template


def _step_id(node: str, revision: int) -> str:
    return f"{node}_r{revision}"


def _snippet(value: Any, limit: int = 80) -> Any:
    """把 args/result 浓缩成前端能快速展示的短摘要。"""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                text = str(v)
                out[k] = text if len(text) <= limit else text[:limit] + "…"
            elif isinstance(v, list):
                if v and all(isinstance(x, (str, int, float)) for x in v):
                    joined = "、".join(str(x) for x in v[:5])
                    out[k] = joined if len(joined) <= limit else joined[:limit] + "…"
                else:
                    out[k] = f"[{len(v)} 项]"
            else:
                out[k] = type(v).__name__
        return out
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    return value


def _execute_events(step_id: str, step_title: str, update: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """execute 节点内部：先按 plan 顺序发 tool_call，tool_trace 回填后发 tool_result。"""
    plan = update.get("plan", []) or []
    traces = update.get("tool_trace", []) or []

    # 先发调用（按 plan 顺序）
    plan_calls = [p for p in plan if p.get("tool") and p.get("status") in ("pending", "done", "failed")]
    for item in plan_calls:
        yield {
            "type": "tool_call",
            "step_id": step_id,
            "tool": item["tool"],
            "args_summary": _snippet(item.get("args", {})),
        }

    # 再发结果（tool_trace 是本次 execute 全部调用的汇总）
    for trace in traces:
        tool = trace.get("tool", "?")
        ok = trace.get("ok", False)
        summary: Any = "成功" if ok else trace.get("error", "失败")
        if ok and trace.get("result"):
            summary = _snippet(trace["result"])
        yield {
            "type": "tool_result",
            "step_id": step_id,
            "tool": tool,
            "ok": ok,
            "summary": summary,
        }


def iter_sse_events(**kwargs) -> Iterator[dict[str, Any]]:
    """把 stream_agent 吐出的 LangGraph chunks 翻译成前端事件。"""
    revision = 0
    final_state: dict[str, Any] | None = None

    # 先给一个「开始」的 status 事件，让前端立即进入 loading
    yield {"type": "status", "phase": "starting", "is_loading": True}

    for chunk in _raw_stream(**kwargs):
        # stream_mode="updates" -> 每个 chunk 形如 {"intake": {...state_dict}}
        for node, update in chunk.items():
            final_state = {**(final_state or {}), **update}

            # revision 来自 state update，plan/reflect 节点都会 bump
            revision = update.get("revision_count", revision) or revision

            step_id = _step_id(node, revision)
            title = _title(node, revision)

            yield {"type": "step_start", "step_id": step_id, "title": title, "revision": revision}

            if node == "execute":
                yield from _execute_events(step_id, title, update)

            key_info = _summarize_node(node, update)
            yield {"type": "step_end", "step_id": step_id, "title": title, "key": key_info}

            # reflect 不通过 → 前端渲染「准备修订」提示
            if node == "reflect":
                passed = (update.get("reflection") or {}).get("passed")
                if passed is False:
                    yield {
                        "type": "note",
                        "text": f"评分 {update.get('reflection', {}).get('score', '?')}，不达标，进入第 {revision + 1} 轮修订",
                    }

    # finalize 节点的 step_end 里不含最终答案，用 answer 事件单独喂给前端
    if final_state:
        yield {"type": "answer", "chunk": final_state.get("final_answer", "")}
        yield {"type": "final", "result": _public_view(final_state)}

    yield {"type": "status", "phase": "done", "is_loading": False}
    yield {"type": "done"}


def _summarize_node(node: str, update: dict[str, Any]) -> dict[str, Any]:
    if node == "intake":
        return {"intent": update.get("intent", "?"), "need_multimodal": update.get("need_multimodal")}
    if node == "collect":
        sources = {}
        for trace in update.get("tool_trace") or []:
            tool = trace.get("tool")
            result = trace.get("result") or {}
            if tool == "parse_health_report":
                sources["report_source"] = result.get("source") if trace.get("ok") else "error"
            elif tool == "detect_ingredients":
                sources["ingredients_source"] = result.get("source") if trace.get("ok") else "error"
        return {"ingredients_count": len(update.get("ingredients", []) or []), **sources}
    if node == "analyze":
        metrics = update.get("health_metrics") or {}
        profile = update.get("profile") or {}
        return {"constraints": profile.get("constraints", []), "metrics_keys": list(metrics.keys())[:5]}
    if node == "plan":
        plan = update.get("plan", []) or []
        return {"planned_tools": [p.get("tool") for p in plan]}
    if node == "reflect":
        return update.get("reflection", {})
    if node == "execute":
        return {
            "ingredients": update.get("ingredients"),
            "knowledge_count": len(update.get("retrieved_knowledge", []) or []),
            "nutrition_total": (update.get("nutrition_report") or {}).get("total"),
        }
    return {}


def _public_view(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": state.get("session_id"),
        "intent": state.get("intent"),
        "profile": state.get("profile"),
        "health_metrics": state.get("health_metrics"),
        "ingredients": state.get("ingredients", []),
        "knowledge_hits": state.get("retrieved_knowledge", []),
        "nutrition": state.get("nutrition_report"),
        "plan": state.get("plan", []),
        "draft_recipe": state.get("draft_recipe"),
        "reflection": state.get("reflection"),
        "revision_count": state.get("revision_count", 0),
        "tool_trace": state.get("tool_trace", []),
        "final_answer": state.get("final_answer", ""),
        "errors": state.get("errors", []),
    }


def iter_sse_lines(**kwargs) -> Iterator[str]:
    """最终喂给 StreamingResponse：每个事件一行 data:，事件之间空行。"""
    for event in iter_sse_events(**kwargs):
        yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
    # heartbeat，避免代理过早断连
    yield ": keep-alive\n\n"
