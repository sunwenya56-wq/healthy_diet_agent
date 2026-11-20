"""集中配置。

设计原则：所有外部依赖（LLM / OCR / YOLO / Milvus / MySQL）都有降级路径，
不配置 key 也能以 Mock 模式跑通完整 Agent 流程，便于演示和测试。
"""

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()           # 自动读取项目根目录 .env，注入 os.environ
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _flag(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # ---- LLM ----
    llm_mock: bool = _flag("LLM_MOCK", True)
    llm_api_key: str = (
        _env("LLM_API_KEY")
        or _env("DEEPSEEK_API_KEY")
        or _env("SILICONFLOW_API_KEY")
        or _env("OPENAI_API_KEY")
    )
    llm_base_url: str = _env("LLM_BASE_URL", "https://api.deepseek.com/v1")
    llm_model: str = _env("LLM_MODEL", "deepseek-chat")
    llm_temperature: float = float(_env("LLM_TEMPERATURE", "0.3"))

    # ---- 多模态 ----
    vision_mock: bool = _flag("VISION_MOCK", True)
    yolo_model_path: str = _env("YOLO_MODEL_PATH", str(BASE_DIR / "models" / "best.pt"))
    ocr_lang: str = _env("OCR_LANG", "chi_sim")
    # --- 混合检测阈值：YOLO 置信度都低于这个值时，自动调 VLM 零样本兜底 ---
    yolo_low_conf_threshold: float = float(_env("YOLO_LOW_CONF_THRESHOLD", "0.6"))
    vlm_ingredient_model: str = _env("VLM_INGREDIENT_MODEL", "deepseek-chat")
    vlm_ingredient_enabled: bool = _flag("VLM_INGREDIENT_ENABLED", True)
    asr_model: str = _env("ASR_MODEL", "XingChenAGI/XingChenGSR-V1.0")
    asr_base_url: str = _env("ASR_BASE_URL", "https://api.siliconflow.cn/v1")
    asr_api_key: str = _env("ASR_API_KEY") or _env("SILICONFLOW_API_KEY")

    # ---- 存储 ----
    database_url: str = _env("DATABASE_URL", f"sqlite:///{(BASE_DIR / 'data' / 'diet.db').as_posix()}")
    vector_backend: str = _env("VECTOR_BACKEND", "memory")  # memory | milvus
    milvus_uri: str = _env("MILVUS_URI", str(BASE_DIR / "data" / "milvus.db"))
    milvus_collection: str = _env("MILVUS_COLLECTION", "diet_guidelines")
    milvus_history_collection: str = _env("MILVUS_HISTORY_COLLECTION", "user_history")
    embedding_dim: int = int(_env("EMBEDDING_DIM", "256"))

    # ---- Agent 行为 ----
    max_revisions: int = int(_env("MAX_REVISIONS", "2"))
    min_quality_score: int = int(_env("MIN_QUALITY_SCORE", "80"))
    max_subtasks: int = int(_env("MAX_SUBTASKS", "5"))

    # ---- API ----
    api_host: str = _env("API_HOST", "127.0.0.1")
    api_port: int = int(_env("API_PORT", "8000"))


settings = Settings()
