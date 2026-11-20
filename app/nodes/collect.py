"""节点 2/3：多模态资源采集 + 健康画像结构化。

OCR / YOLO 都通过工具注册表调用（与 MCP 暴露的是同一套工具），
结构化抽取用正则兜底、LLM 增强，保证没有模型时也能得到稳定结果。
"""

import re
from typing import Any

from app.adapters.llm import LLMClient, LLMError
from app.state import DietState
from app.tools.registry import ToolRegistry

_llm = LLMClient()

_PATTERNS = {
    "bmi": r"BMI\s*[:：]?\s*(\d+(?:\.\d+)?)",
    "height_cm": r"身高\s*[:：]?\s*(\d+(?:\.\d+)?)\s*cm",
    "weight_kg": r"体重\s*[:：]?\s*(\d+(?:\.\d+)?)\s*kg",
    "fasting_glucose": r"空腹血糖\s*[:：]?\s*(\d+(?:\.\d+)?)",
    "hba1c": r"糖化血红蛋白\s*[:：]?\s*(\d+(?:\.\d+)?)",
    "total_cholesterol": r"总胆固醇\s*[:：]?\s*(\d+(?:\.\d+)?)",
    "triglyceride": r"甘油三酯\s*[:：]?\s*(\d+(?:\.\d+)?)",
    "ldl": r"低密度脂蛋白\s*[:：]?\s*(\d+(?:\.\d+)?)",
    "systolic": r"血压\s*[:：]?\s*(\d+)\s*/\s*(\d+)",
    "allergy": r"过敏史\s*[:：]?\s*([^\s，,。;；]+)",
}


def extract_metrics(report_text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key, pattern in _PATTERNS.items():
        match = re.search(pattern, report_text)
        if not match:
            continue
        if key == "systolic":
            metrics["systolic"] = float(match.group(1))
            metrics["diastolic"] = float(match.group(2))
        elif key == "allergy":
            value = match.group(1)
            metrics["allergy"] = [] if value in {"无", "未见", "None"} else re.split(r"[、和及]", value)
        else:
            metrics[key] = float(match.group(1))
    return metrics


def derive_constraints(metrics: dict[str, Any]) -> list[str]:
    """把医学指标翻译成菜谱要遵守的饮食约束——这是「个性化」的真正来源。"""
    constraints: list[str] = []

    glucose = metrics.get("fasting_glucose")
    if (glucose and glucose >= 6.1) or (metrics.get("hba1c") or 0) >= 6.0:
        constraints.append("控糖")

    chol = metrics.get("total_cholesterol") or 0
    ldl = metrics.get("ldl") or 0
    triglyceride = metrics.get("triglyceride") or 0
    if chol >= 5.2 or ldl >= 3.4 or triglyceride >= 1.7:
        constraints.append("低脂")

    if (metrics.get("systolic") or 0) >= 140 or (metrics.get("diastolic") or 0) >= 90:
        constraints.append("低盐")

    bmi = metrics.get("bmi") or 0
    if bmi >= 24:
        constraints.append("控热量")

    for allergen in metrics.get("allergy") or []:
        constraints.append(f"避敏:{allergen}")

    return constraints


def collect_node(state: DietState, registry: ToolRegistry) -> dict[str, Any]:
    """调用 OCR / YOLO 工具收集原始素材。"""
    updates: dict[str, Any] = {}
    errors: list[str] = []
    traces: list[dict[str, Any]] = []

    if state.get("report_image_path"):
        out = registry.call("parse_health_report", {"image_path": state["report_image_path"]})
        traces.append(out)
        if out["ok"]:
            updates["report_text"] = out["result"]["text"]
        else:
            errors.append(out["error"])

    if state.get("fridge_image_path"):
        out = registry.call("detect_ingredients", {"image_path": state["fridge_image_path"]})
        traces.append(out)
        if out["ok"]:
            updates["ingredients"] = out["result"]["ingredients"]
        else:
            errors.append(out["error"])

    if traces:
        updates["tool_trace"] = traces
    if errors:
        updates["errors"] = errors
    return updates


def analyze_node(state: DietState) -> dict[str, Any]:
    """把报告原文转成 profile / constraints。"""
    report_text = state.get("report_text", "")
    if not report_text:
        return {"profile": state.get("profile", {}), "health_metrics": state.get("health_metrics", {})}

    metrics = extract_metrics(report_text)

    if _llm.available:
        try:
            enriched = _llm.chat_json(
                "从体检报告抽取指标，输出 JSON，字段 bmi/weight_kg/fasting_glucose/total_cholesterol/"
                f"systolic/diastolic/allergy(list)。报告：{report_text}",
                system="你是医学报告结构化解析器，只输出 JSON。",
            )
            metrics = {**metrics, **{k: v for k, v in enriched.items() if v not in (None, "", [])}}
        except LLMError:
            pass

    constraints = derive_constraints(metrics)
    profile = {
        "constraints": constraints,
        "bmi": metrics.get("bmi"),
        "allergy": metrics.get("allergy", []),
        "summary": "、".join(constraints) if constraints else "无特殊限制",
    }
    return {"health_metrics": metrics, "profile": profile}
