"""向量检索适配器：Milvus（生产）→ 内存混合检索（本地降级）。

职责分两块：
    1. 膳食指南检索 —— 全局知识，collection=`diet_guidelines`
    2. 用户历史交互 —— 个性化记忆，collection=`user_history`

设计原则：所有 Milvus 操作都降级，连不上就打日志 return，
绝不让向量库故障影响 Agent 主流程（参考 FastAPI + Milvus 的容错模式）。
"""

from __future__ import annotations

import logging
import math
import re
import uuid
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

_STOPWORDS = {"的", "了", "和", "与", "及", "在", "是", "为", "对", "与", "或"}


def embed(text: str) -> list[float]:
    """字符 bigram + 关键词哈希到固定维度，L2 归一化。"""
    dim = settings.embedding_dim
    vec = [0.0] * dim
    text = re.sub(r"\s+", "", text)
    grams = [text[i : i + 2] for i in range(max(len(text) - 1, 1))]
    grams += [w for w in re.findall(r"[A-Za-z]+|\d+", text)]
    for gram in grams:
        if not gram:
            continue
        h = hash(gram)
        vec[h % dim] += 1.0
        vec[(h // dim) % dim] += 0.5
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _keywords(text: str) -> set[str]:
    tokens = set(re.findall(r"[A-Za-z]+|\d+", text))
    tokens |= {t for t in re.split(r"[，。、；：\s,.;:()（）]+", text) if t}
    return {t for t in tokens if len(t) >= 2 and t not in _STOPWORDS}


def hybrid_score(query: str, doc: str) -> float:
    """0.7 向量余弦 + 0.3 关键词覆盖率。"""
    vec_score = _cosine(embed(query), embed(doc))
    q_keys = _keywords(query)
    doc_keys = _keywords(doc)
    keyword = len(q_keys & doc_keys) / len(q_keys) if q_keys else 0.0
    return round(0.7 * vec_score + 0.3 * keyword, 4)


class InMemoryVectorStore:
    backend = "memory"

    def __init__(self) -> None:
        self._docs: list[dict[str, Any]] = []

    def add(self, docs: list[dict[str, Any]]) -> None:
        self._docs.extend(docs)

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        scored = [(hybrid_score(query, d["text"]), d) for d in self._docs]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{**d, "score": score} for score, d in scored[:top_k]]


class MilvusVectorStore:
    backend = "milvus"

    def __init__(self) -> None:
        from pymilvus import MilvusClient

        self._client = MilvusClient(uri=settings.milvus_uri)
        self._name = settings.milvus_collection

    def add(self, docs: list[dict[str, Any]]) -> None:
        if not self._client.has_collection(self._name):
            self._client.create_collection(
                collection_name=self._name,
                dimension=settings.embedding_dim,
                auto_id=True,
            )
        rows = [
            {"vector": embed(d["text"]), "text": d["text"], "title": d.get("title", "")}
            for d in docs
        ]
        if rows:
            self._client.insert(collection_name=self._name, data=rows)

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        hits = self._client.search(
            collection_name=self._name,
            data=[embed(query)],
            limit=top_k,
            output_fields=["text", "title"],
        )[0]
        return [
            {
                "text": h["entity"]["text"],
                "title": h["entity"].get("title", ""),
                "score": round(float(h["distance"]), 4),
            }
            for h in hits
        ]


def load_guidelines(directory: Path) -> list[dict[str, Any]]:
    docs = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for block in re.split(r"\n\s*\n", text):
            block = block.strip()
            if len(block) < 20:
                continue
            docs.append({"title": path.stem, "text": block})
    return docs


def build_store() -> InMemoryVectorStore | MilvusVectorStore:
    docs = load_guidelines(Path(__file__).resolve().parent.parent.parent / "data" / "guidelines")
    if settings.vector_backend == "milvus":
        try:
            store = MilvusVectorStore()
            store.add(docs)
            return store
        except Exception:
            pass  # 连不上 Milvus 就降级，不让整个 Agent 挂掉

    store = InMemoryVectorStore()
    store.add(docs)
    return store


# ---------------------------------------------------------------------------
# 用户历史交互向量化记忆
# ---------------------------------------------------------------------------

_history_memory: list[dict[str, Any]] = []


def _get_milvus_client():
    """懒加载 Milvus client，连不上返回 None（不抛异常）。"""
    if settings.vector_backend != "milvus":
        return None
    try:
        from pymilvus import MilvusClient

        return MilvusClient(uri=settings.milvus_uri)
    except Exception as exc:
        logger.warning("Milvus client 初始化失败: %s: %s", type(exc).__name__, exc)
        return None


def store_user_interaction(
    user_id: int,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """把一次 Agent 交互的 final_answer 存入用户历史记忆。

    生产走 Milvus（`user_history` collection），本地降级走内存列表，
    无论走哪条路径都不影响主流程。
    """
    if not text or not text.strip():
        return False

    meta = metadata or {}
    doc = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "text": text.strip(),
        "query": meta.get("query", ""),
        "intent": meta.get("intent", ""),
        "session_id": meta.get("session_id", ""),
    }

    client = _get_milvus_client()
    if client is not None:
        try:
            name = settings.milvus_history_collection
            if not client.has_collection(name):
                client.create_collection(
                    collection_name=name,
                    dimension=settings.embedding_dim,
                    auto_id=False,
                )
            client.insert(
                collection_name=name,
                data=[{
                    "id": doc["id"],
                    "vector": embed(doc["text"]),
                    "text": doc["text"],
                    "user_id": user_id,
                    "query": doc["query"],
                    "intent": doc["intent"],
                    "session_id": doc["session_id"],
                }],
            )
            return True
        except Exception as exc:
            logger.warning("Milvus 用户历史写入失败，降级到内存: %s: %s", type(exc).__name__, exc)

    # 内存降级
    _history_memory.append(doc)
    if len(_history_memory) > 200:
        _history_memory.pop(0)
    return True


def search_user_history(query: str, user_id: int | None = None, top_k: int = 3) -> list[dict[str, Any]]:
    """检索用户历史交互，返回相似度最高的 top_k 条 final_answer。

    用于 Planner/Reflector 做 query 改写或个性化参考。
    """
    client = _get_milvus_client()
    if client is not None:
        try:
            name = settings.milvus_history_collection
            if not client.has_collection(name):
                return []
            filter_expr = f'user_id == {user_id}' if user_id is not None else None
            hits = client.search(
                collection_name=name,
                data=[embed(query)],
                limit=top_k,
                filter=filter_expr,
                output_fields=["text", "query", "intent", "user_id"],
            )[0]
            return [
                {
                    "text": h["entity"]["text"],
                    "query": h["entity"].get("query", ""),
                    "score": round(float(h["distance"]), 4),
                }
                for h in hits
            ]
        except Exception as exc:
            logger.warning("Milvus 用户历史检索失败，降级到内存: %s: %s", type(exc).__name__, exc)

    # 内存降级：线性打分
    pooled = _history_memory if user_id is None else [d for d in _history_memory if d["user_id"] == user_id]
    scored = [(hybrid_score(query, d["text"]), d) for d in pooled]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"text": d["text"], "query": d["query"], "score": round(s, 4)} for s, d in scored[:top_k]]

