"""节点 1：意图识别。

先规则、后 LLM——规则能覆盖 90% 的中文短句，又快又稳定；
只有规则判不出来的模糊表达才交给模型，避免每次请求都多一次 LLM 调用。
"""

import re
from typing import Any

from app.adapters.llm import LLMClient, LLMError
from app.state import DietState

_llm = LLMClient()

_RECIPE = re.compile(r"菜谱|食谱|做饭|吃什么|晚餐|早餐|午餐|推荐.*餐|搭配")
_REPORT = re.compile(r"体检|报告|指标|化验|血压|血糖")
_NUTRITION = re.compile(r"热量|营养|能不能吃|适合吃|卡路里|蛋白质|减脂")


def classify_intent(query: str) -> str:
    """按优先级匹配：有明确的「做饭」诉求时，报告只是输入材料而不是目的。"""
    if _RECIPE.search(query):
        return "generate_recipe"
    if _REPORT.search(query):
        return "parse_report"
    if _NUTRITION.search(query):
        return "nutrition_consult"
    return "chitchat"


def intake_node(state: DietState) -> dict[str, Any]:
    query = state.get("user_query", "").strip()

    if _llm.available and len(query) < 8:
        try:
            parsed = _llm.chat_json(
                f"判断用户意图，只返回 JSON：{{\"intent\": \"generate_recipe|nutrition_consult|parse_report|chitchat\"}}。用户：{query}",
                system="你是意图分类器，只输出 JSON。",
            )
            intent = parsed.get("intent", "chitchat")
        except (LLMError, KeyError):
            intent = classify_intent(query)
    else:
        intent = classify_intent(query)

    need_multimodal = bool(state.get("report_image_path") or state.get("fridge_image_path"))
    return {
        "intent": intent,
        "need_multimodal": need_multimodal,
        "revision_count": state.get("revision_count", 0),
    }


def route_after_intake(state: DietState) -> str:
    """意图直接决定走资源采集还是直接进入规划。"""
    return "collect" if state.get("need_multimodal") else "plan"
