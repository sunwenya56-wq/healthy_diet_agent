"""MCP 风格的工具注册表。

每个工具都是一段「可被 Agent 发现 + 可被外部（MCP Client / LLM Function Calling）调用」
的独立能力：有 JSON Schema、有统一 handler 签名、有结构化返回值。
老代码把这些能力直接写死在 Flask 路由里，无法复用也无法被模型自主选择。
"""

from dataclasses import dataclass, field
from typing import Any, Callable

ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    tags: list[str] = field(default_factory=list)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"工具重复注册: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"未注册的工具: {name}")
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """按 MCP tools/list 的格式输出，可直接暴露给 MCP Server。"""
        return [
            {"name": s.name, "description": s.description, "inputSchema": s.input_schema}
            for s in self._tools.values()
        ]

    def call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """统一执行入口：异常转成结构化错误，避免一个工具失败拖垮整张图。"""
        args = args or {}
        try:
            spec = self.get(name)
            result = spec.handler(args)
            return {"ok": True, "tool": name, "args": args, "result": result}
        except Exception as exc:  # noqa: BLE001 - 工具边界需要兜底
            return {"ok": False, "tool": name, "args": args, "error": f"{type(exc).__name__}: {exc}"}
