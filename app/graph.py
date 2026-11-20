"""LangGraph 图构建：状态机 + 条件边 + 反思回环 + 会话级 checkpointer。"""

import uuid
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.nodes.collect import analyze_node, collect_node
from app.nodes.execute import execute_node
from app.nodes.intake import intake_node, route_after_intake
from app.nodes.plan import plan_node
from app.nodes.reflect import finalize_node, reflect_node, route_after_reflect
from app.state import DietState
from app.tools.implementations import build_registry
from app.tools.registry import ToolRegistry


def build_graph(registry: ToolRegistry | None = None, checkpointer: Any | None = None):
    registry = registry or build_registry()

    graph = StateGraph(DietState)

    graph.add_node("intake", intake_node)
    graph.add_node("collect", lambda state: collect_node(state, registry))
    graph.add_node("analyze", analyze_node)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", lambda state: execute_node(state, registry))
    graph.add_node("reflect", reflect_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", route_after_intake, {"collect": "collect", "plan": "plan"})
    graph.add_edge("collect", "analyze")
    graph.add_edge("analyze", "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "reflect")
    graph.add_conditional_edges("reflect", route_after_reflect, {"plan": "plan", "finalize": "finalize"})
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


_GRAPH = None


def get_graph():
    """进程级单例：checkpointer 需要复用，否则会话状态会丢。"""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_agent(
    query: str,
    user_id: int = 1,
    session_id: str | None = None,
    report_image_path: str | None = None,
    fridge_image_path: str | None = None,
    ingredients: list[str] | None = None,
    graph=None,
) -> dict[str, Any]:
    session_id = session_id or uuid.uuid4().hex
    graph = graph or get_graph()

    initial: dict[str, Any] = {
        "session_id": session_id,
        "user_id": user_id,
        "user_query": query,
        "report_image_path": report_image_path,
        "fridge_image_path": fridge_image_path,
        "revision_count": 0,
        "tool_trace": [],
        "errors": [],
    }
    if ingredients:
        initial["ingredients"] = ingredients

    config = {"configurable": {"thread_id": session_id}}
    return graph.invoke(initial, config=config)


def stream_agent(
    query: str,
    user_id: int = 1,
    session_id: str | None = None,
    report_image_path: str | None = None,
    fridge_image_path: str | None = None,
    ingredients: list[str] | None = None,
    graph=None,
):
    """按节点增量输出，供 SSE / WebSocket 前端做「思考过程」实时展示。"""
    session_id = session_id or uuid.uuid4().hex
    graph = graph or get_graph()
    initial: dict[str, Any] = {
        "session_id": session_id,
        "user_id": user_id,
        "user_query": query,
        "report_image_path": report_image_path,
        "fridge_image_path": fridge_image_path,
        "revision_count": 0,
        "tool_trace": [],
        "errors": [],
    }
    if ingredients:
        initial["ingredients"] = ingredients

    config = {"configurable": {"thread_id": session_id}}
    yield from graph.stream(initial, config=config, stream_mode="updates")
