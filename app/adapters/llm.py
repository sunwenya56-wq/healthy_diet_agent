"""LLM 适配器：OpenAI 兼容协议（DeepSeek / 硅基流动 / vLLM 均可接）。"""

import json
import re
from typing import Any

from app.config import settings


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self) -> None:
        self._client = None
        if not settings.llm_mock and settings.llm_api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=settings.llm_api_key,
                    base_url=settings.llm_base_url,
                    timeout=60.0,
                )
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def chat_text(self, prompt: str, system: str = "") -> str:
        if not self.available:
            raise LLMError("LLM 未配置（LLM_MOCK=1 或缺少 API Key）")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=settings.llm_temperature,
        )
        return resp.choices[0].message.content or ""

    def chat_json(self, prompt: str, system: str = "") -> dict[str, Any]:
        """要求模型返回 JSON，并从可能包裹的 markdown 代码块中稳健解析。"""
        raw = self.chat_text(prompt, system=system)
        return _loads_json(raw)


def _loads_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"LLM 返回不是合法 JSON: {text[:200]}") from exc


llm = LLMClient()
