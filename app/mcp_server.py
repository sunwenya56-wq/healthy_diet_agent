"""把 Agent 的 5 个工具暴露成标准 MCP Server（stdio）。

启动：python -m app.mcp_server
之后任意 MCP Client（Claude Desktop / Dify / 自研调度器）都能发现并调用这些工具，
这就是简历里「低代码平台 + 高代码 Agent 共用同一套工具协议」的具体做法。
"""

from mcp.server.fastmcp import FastMCP

from app.tools.implementations import (
    tool_calculate_nutrition,
    tool_detect_ingredients,
    tool_generate_recipe,
    tool_parse_report,
    tool_search_guide,
)

mcp = FastMCP("healthy-diet-agent")


@mcp.tool()
def parse_health_report(image_path: str) -> dict:
    """OCR / PDF 解析体检报告，返回报告原文。"""
    return tool_parse_report({"image_path": image_path})


@mcp.tool()
def detect_ingredients(image_path: str, min_confidence: float = 0.5) -> dict:
    """YOLO 检测冰箱图片中的食材。"""
    return tool_detect_ingredients({"image_path": image_path, "min_confidence": min_confidence})


@mcp.tool()
def search_diet_guide(query: str, top_k: int = 3) -> dict:
    """检索《中国居民膳食指南》知识库。"""
    return tool_search_guide({"query": query, "top_k": top_k})


@mcp.tool()
def calculate_nutrition(ingredients: list[str], portion_g: float = 150) -> dict:
    """按食材清单估算热量与三大营养素。"""
    return tool_calculate_nutrition({"ingredients": ingredients, "portion_g": portion_g})


@mcp.tool()
def generate_recipe(ingredients: list[str], constraints: list[str] | None = None) -> dict:
    """结合健康约束与现有食材生成结构化菜谱。"""
    return tool_generate_recipe({"ingredients": ingredients, "constraints": constraints or []})


if __name__ == "__main__":
    mcp.run()
