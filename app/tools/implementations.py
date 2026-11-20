"""工具实现：OCR / 食材检测 / 指南检索 / 营养计算 / 菜谱生成。"""

from typing import Any

from app.adapters.llm import LLMClient, LLMError
from app.adapters.vector import build_store
from app.adapters.vision import detect_ingredients, parse_health_report
from app.config import settings
from app.tools.registry import ToolRegistry, ToolSpec

# 每 100g 的营养成分：(热量 kcal, 蛋白质 g, 脂肪 g, 碳水 g, 钠 mg)
NUTRITION_TABLE: dict[str, tuple[float, float, float, float, float]] = {
    "西红柿": (20, 0.9, 0.2, 4.0, 5),
    "鸡蛋": (143, 13.0, 9.5, 1.1, 130),
    "青椒": (22, 1.0, 0.2, 5.0, 3),
    "土豆": (77, 2.0, 0.2, 17.0, 6),
    "鸡胸肉": (133, 24.0, 5.0, 0.0, 60),
    "西兰花": (34, 2.8, 0.4, 7.0, 18),
    "胡萝卜": (41, 0.9, 0.2, 9.6, 70),
    "洋葱": (40, 1.1, 0.2, 9.0, 4),
    "鲈鱼": (105, 18.6, 3.4, 0.0, 144),
    "豆腐": (82, 8.1, 4.2, 3.0, 7),
    "菠菜": (28, 2.6, 0.3, 4.5, 85),
    "香菇": (26, 2.2, 0.3, 5.0, 1),
    "牛肉": (125, 20.0, 4.0, 2.0, 84),
    "芹菜": (16, 0.8, 0.1, 3.0, 159),
    "虾仁": (93, 18.6, 0.8, 2.6, 165),
    "冬瓜": (12, 0.4, 0.2, 2.6, 1),
    "木耳": (27, 1.5, 0.2, 6.0, 8),
    "青菜": (15, 1.5, 0.3, 2.0, 73),
}
DEFAULT_PORTION_G = 150
_UNKNOWN = (80, 5.0, 2.0, 10.0, 40)

_llm = LLMClient()
_store = None


def _vector_store():
    global _store
    if _store is None:
        _store = build_store()
    return _store


# ---------------------------------------------------------------- 工具实现


def tool_parse_report(args: dict[str, Any]) -> dict[str, Any]:
    text, source = parse_health_report(args["image_path"])
    return {"text": text, "source": source}


def tool_detect_ingredients(args: dict[str, Any]) -> dict[str, Any]:
    result = detect_ingredients(args["image_path"])
    raw = result["ingredients"]  # list[{name, confidence}]
    names = [d["name"] for d in raw if d["confidence"] >= args.get("min_confidence", 0.5)]
    return {
        "ingredients": names,
        "raw": raw,
        "source": result["source"],   # "yolo" / "vlm" / "mock" —— 可观测性
        "fallback": result["fallback"],
    }


def tool_search_guide(args: dict[str, Any]) -> dict[str, Any]:
    query = args["query"]
    docs = _vector_store().search(query, top_k=args.get("top_k", 3))
    return {"backend": _vector_store().backend, "hits": docs}


def tool_calculate_nutrition(args: dict[str, Any]) -> dict[str, Any]:
    ingredients = args.get("ingredients", [])
    portion = float(args.get("portion_g", DEFAULT_PORTION_G)) / 100.0
    items = []
    for name in ingredients:
        kcal, protein, fat, carb, sodium = NUTRITION_TABLE.get(name, _UNKNOWN)
        items.append(
            {
                "name": name,
                "portion_g": portion * 100,
                "kcal": round(kcal * portion, 1),
                "protein_g": round(protein * portion, 1),
                "fat_g": round(fat * portion, 1),
                "carb_g": round(carb * portion, 1),
                "sodium_mg": round(sodium * portion, 1),
            }
        )
    total = {k: round(sum(i[k] for i in items), 1) for k in ("kcal", "protein_g", "fat_g", "carb_g", "sodium_mg")}
    return {"items": items, "total": total}


def tool_generate_recipe(args: dict[str, Any]) -> dict[str, Any]:
    ingredients = args.get("ingredients") or ["西红柿", "鸡蛋"]
    constraints = args.get("constraints") or []
    revision = int(args.get("revision", 0))
    hint = args.get("hint", "")

    if _llm.available:
        try:
            return _llm.chat_json(_recipe_prompt(ingredients, constraints, hint), system=_RECIPE_SYSTEM)
        except LLMError:
            pass

    return _mock_recipe(ingredients, constraints, revision)


_RECIPE_SYSTEM = (
    "你是注册营养师。只输出 JSON，字段：title, ingredients(list), steps(list), "
    "nutrition_note, constraint_notes(list), est_kcal。不要输出多余文字。"
)


def _recipe_prompt(ingredients: list[str], constraints: list[str], hint: str) -> str:
    return (
        f"可用食材：{', '.join(ingredients)}\n"
        f"饮食约束：{', '.join(constraints) or '无'}\n"
        f"上一轮评审意见：{hint or '无'}\n"
        "请生成一份可落地的家常菜谱，必须使用至少两种现有食材。"
    )


def _mock_recipe(ingredients: list[str], constraints: list[str], revision: int) -> dict[str, Any]:
    """Mock 生成器：第一版故意留下可被 Reflector 发现的缺口，修订轮补齐。"""
    main = ingredients[:2] if len(ingredients) >= 2 else ["西红柿", "鸡蛋"]
    title = f"{main[0]}{main[1]}快手餐" if len(main) == 2 else f"{main[0]}家常餐"

    if revision == 0:
        steps = [f"1. {main[0]}洗净切块", f"2. 热锅下{main[1]}翻炒"]
        note, notes = "", []
    else:
        steps = [
            f"1. {main[0]}洗净切块，{main[1]}预处理备用",
            f"2. 少油热锅，下{main[1]}翻炒至变色",
            f"3. 加入{main[0]}与其余食材，中火翻炒 3 分钟",
            "4. 调味出锅，全程控盐控油",
        ]
        note = "本菜蛋白质充足、膳食纤维适中，建议搭配一份粗粮主食。"
        notes = [f"已按「{c}」调整" for c in constraints]

    return {
        "title": title,
        "ingredients": main,
        "steps": steps,
        "nutrition_note": note,
        "constraint_notes": notes,
        "est_kcal": 420 if revision else 560,
    }


# ---------------------------------------------------------------- 注册


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="parse_health_report",
            description="OCR/PDF 解析体检报告，返回原文",
            input_schema={
                "type": "object",
                "properties": {"image_path": {"type": "string", "description": "报告图片或 PDF 路径"}},
                "required": ["image_path"],
            },
            handler=tool_parse_report,
            tags=["multimodal", "ocr"],
        )
    )
    registry.register(
        ToolSpec(
            name="detect_ingredients",
            description="YOLO 检测冰箱图片中的食材",
            input_schema={
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"},
                    "min_confidence": {"type": "number", "default": 0.5},
                },
                "required": ["image_path"],
            },
            handler=tool_detect_ingredients,
            tags=["multimodal", "vision"],
        )
    )
    registry.register(
        ToolSpec(
            name="search_diet_guide",
            description="检索《中国居民膳食指南》知识库（向量+关键词混合检索）",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3},
                },
                "required": ["query"],
            },
            handler=tool_search_guide,
            tags=["rag"],
        )
    )
    registry.register(
        ToolSpec(
            name="calculate_nutrition",
            description="按食材清单估算热量与三大营养素",
            input_schema={
                "type": "object",
                "properties": {
                    "ingredients": {"type": "array", "items": {"type": "string"}},
                    "portion_g": {"type": "number", "default": DEFAULT_PORTION_G},
                },
                "required": ["ingredients"],
            },
            handler=tool_calculate_nutrition,
            tags=["nutrition"],
        )
    )
    registry.register(
        ToolSpec(
            name="generate_recipe",
            description="结合健康约束与现有食材生成结构化菜谱",
            input_schema={
                "type": "object",
                "properties": {
                    "ingredients": {"type": "array", "items": {"type": "string"}},
                    "constraints": {"type": "array", "items": {"type": "string"}},
                    "revision": {"type": "integer", "default": 0},
                    "hint": {"type": "string", "default": ""},
                },
                "required": ["ingredients"],
            },
            handler=tool_generate_recipe,
            tags=["llm"],
        )
    )
    return registry
