"""视觉适配器：体检报告 OCR + 冰箱食材混合检测。

食材检测是三段式混合流水线：
    1. YOLO（监督训练）—— 主路径，已知食材高精度
    2. VLM（零样本多模态）—— 兜底路径，YOLO 全置信度 < threshold 时自动触发，覆盖长尾食材
    3. Mock（按文件名 hash 稳定样例）—— 终极兜底，保证 Demo 永远能跑

体检报告走两条真实路径：
    1. OCR / PDF 文本抽取（快、免费）
    2. OCR 不可用或识别质量不达标时，回落到 VLM 逐字转录
真实模式（VISION_MOCK=0）下绝不伪造数据：两条路径都失败就抛错，交给上层记录。
Mock 只在 VISION_MOCK=1（纯 Demo）时启用。
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


class ReportParseError(RuntimeError):
    """体检报告解析失败（OCR 与 VLM 都没拿到可用文本）。"""

MOCK_REPORTS = [
    "体检报告 姓名:张伟 性别:男 年龄:32\n身高175cm 体重68kg BMI22.2\n空腹血糖5.1mmol/L 总胆固醇4.2mmol/L 甘油三酯1.1mmol/L\n血压118/76mmHg 过敏史:无",
    "体检报告 姓名:李婷 性别:女 年龄:45\n身高162cm 体重70kg BMI26.7\n空腹血糖6.4mmol/L 糖化血红蛋白6.3% 总胆固醇5.1mmol/L\n血压126/82mmHg 过敏史:无",
    "体检报告 姓名:王强 性别:男 年龄:40\n身高172cm 体重85kg BMI28.7\n空腹血糖5.6mmol/L 总胆固醇6.4mmol/L 低密度脂蛋白4.1mmol/L\n血压132/88mmHg 过敏史:无",
    "体检报告 姓名:陈晨 性别:女 年龄:26\n身高165cm 体重52kg BMI19.1\n空腹血糖4.8mmol/L 总胆固醇3.9mmol/L\n血压110/70mmHg 过敏史:花生、海鲜",
    "体检报告 姓名:赵建国 性别:男 年龄:68\n身高168cm 体重74kg BMI26.2\n空腹血糖6.1mmol/L 总胆固醇5.5mmol/L\n血压152/94mmHg 过敏史:无",
]

MOCK_INGREDIENTS = [
    ["西红柿", "鸡蛋", "青椒", "土豆"],
    ["鸡胸肉", "西兰花", "胡萝卜", "洋葱"],
    ["鲈鱼", "豆腐", "菠菜", "香菇"],
    ["牛肉", "芹菜", "西红柿", "鸡蛋"],
    ["虾仁", "冬瓜", "木耳", "青菜"],
]


def _pick(samples: list, path: str) -> Any:
    """按文件名稳定取样例，避免同一张图每次结果不同。"""
    digest = hashlib.md5(Path(path).name.encode("utf-8")).hexdigest()
    return samples[int(digest, 16) % len(samples)]


# ---------------------------------------------------------------------------
# 体检报告解析：OCR / PDF → VLM 转录兜底
# ---------------------------------------------------------------------------

_REPORT_KEYWORDS = ("体检", "身高", "体重", "BMI", "血糖", "胆固醇", "血压", "报告")


def _looks_like_report(text: str) -> bool:
    """OCR 输出是否包含可用的体检指标，防止把乱码当成结果。"""
    if not text or len(text.strip()) < 10:
        return False
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    if cjk < 5:
        return False
    return any(keyword in text for keyword in _REPORT_KEYWORDS)


def parse_health_report(image_path: str) -> tuple[str, str]:
    """返回 (报告文本, 来源)。来源取值：pdf / ocr / vlm / mock。

    VISION_MOCK=0 时：OCR 与 VLM 都失败会抛 ReportParseError，绝不返回假报告。
    """
    suffix = Path(image_path).suffix.lower()

    if not settings.vision_mock:
        errors: list[str] = []

        if suffix == ".pdf":
            try:
                text = _pdf_text(image_path)
                if _looks_like_report(text):
                    return text, "pdf"
                errors.append("PDF 未抽取到有效体检指标")
            except Exception as exc:
                errors.append(f"PDF 抽取失败 {type(exc).__name__}: {exc}")
        else:
            try:
                text = _ocr_text(image_path)
                if _looks_like_report(text):
                    return text, "ocr"
                errors.append("OCR 未识别到有效体检指标")
            except Exception as exc:
                errors.append(f"OCR 不可用 {type(exc).__name__}: {exc}")

        text = _try_vlm_report(image_path)
        if text:
            return text, "vlm"

        errors.append("VLM 转录失败")
        logger.warning("体检报告解析失败 %s: %s", image_path, "；".join(errors))
        raise ReportParseError("体检报告解析失败：" + "；".join(errors))

    return str(_pick(MOCK_REPORTS, image_path)), "mock"


def _pdf_text(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader

    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _ocr_text(path: str) -> str:
    from PIL import Image
    import pytesseract

    return pytesseract.image_to_string(Image.open(path), lang=settings.ocr_lang)


# ---------------------------------------------------------------------------
# 冰箱食材混合检测：YOLO → VLM → Mock
# ---------------------------------------------------------------------------

def detect_ingredients(image_path: str) -> dict[str, Any]:
    """混合检测主入口。返回结构化结果（含 source 字段，上层可观测）。

    三段式：
        1. YOLO 主路径（置信度均 ≥ threshold → 直接收工）
        2. VLM 兜底（YOLO 置信度全 < threshold 或 YOLO 不可用时 → 零样本）
        3. Mock 兜底（连 VLM 都没配时 → 文件名 hash 稳定样例）
    """
    threshold = settings.yolo_low_conf_threshold

    # —— 路径 1：真实 YOLO ——
    yolo_result = _try_yolo(image_path)
    if yolo_result is not None:
        detections = yolo_result["detections"]
        confs = [d["confidence"] for d in detections]
        if detections and max(confs) >= threshold:
            # YOLO 置信度够高，直接收工
            return {"source": "yolo", "ingredients": detections, "fallback": None}

        # YOLO 输出了但全是低置信度 → 记录下来，尝试 VLM 补充
        yolo_low = detections
    else:
        yolo_low = []

    # —— 路径 2：VLM 多模态兜底 ——
    if settings.vlm_ingredient_enabled:
        vlm_result = _try_vlm_ingredients(image_path)
        if vlm_result is not None:
            return {"source": "vlm", "ingredients": vlm_result, "fallback": "yolo_low"}

    # —— 路径 3：Mock 终极兜底 ——
    names = _pick(MOCK_INGREDIENTS, image_path)
    detections = [{"name": name, "confidence": 0.88} for name in names]
    return {"source": "mock", "ingredients": detections, "fallback": None}


def _try_yolo(path: str) -> dict[str, Any] | None:
    """尝试 YOLO 检测。返回 None 表示依赖/权重缺失或异常。"""
    if settings.vision_mock:
        return None
    try:
        from ultralytics import YOLO

        model = _load_yolo()
        result = model(path)[0]
        detections = []
        for box in result.boxes:
            detections.append(
                {
                    "name": result.names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 3),
                }
            )
        return {"detections": detections}
    except Exception:
        return None


_YOLO_CACHE = {}


def _load_yolo():
    """模型只加载一次，避免每个请求都重新初始化权重。"""
    if "model" not in _YOLO_CACHE:
        from ultralytics import YOLO

        _YOLO_CACHE["model"] = YOLO(settings.yolo_model_path)
    return _YOLO_CACHE["model"]


def _try_vlm_report(path: str) -> str | None:
    """OCR 不可用时的兜底：让多模态模型直接逐字转录体检报告。"""
    system = (
        "你是医学报告 OCR 助手。逐字转录图中所有可见文字，"
        "保持「指标名 → 数值」的对应关系，不要推测或补全图中不存在的内容。"
    )
    prompt = "请转录这张体检报告图片中的全部文字，逐行输出，不要添加解释。"
    return _vlm_vision(path, system, prompt)


def _try_vlm_ingredients(path: str) -> list[dict] | None:
    """用多模态 chat 零样本识别食材。

    零样本 = 不需要训练，直接问模型「图里有什么食材」——适合长尾类别兜底。
    模型自动复用你 .env 里已经配好的 key。
    """
    system = (
        "你是专业的食材识别助手。只列出图中清晰可见的完整食材名称，"
        "不要输出食材状态（如切好的/生的/熟的）或数量。"
    )
    prompt = (
        "请识别图片里的所有食材，返回严格的 JSON 数组，"
        "每个元素形如 {\"name\": \"食材名\", \"confidence\": 0.0~1.0}。"
        "不要返回任何 JSON 以外的文字。"
    )
    raw = _vlm_vision(path, system, prompt)
    if raw is None:
        return None

    try:
        # 稳健 JSON 解析（防模型 markdown 包裹）
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        if text.startswith("["):
            data = json.loads(text)
        else:
            arr = text.find("[")
            arr_end = text.rfind("]")
            if arr == -1 or arr_end == -1:
                raise ValueError(f"VLM 未返回数组: {raw[:200]}")
            data = json.loads(text[arr : arr_end + 1])

        out = []
        for item in data:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            conf = float(item.get("confidence", 0.7))
            out.append({"name": name, "confidence": round(min(max(conf, 0.1), 1.0), 3)})
        return out
    except Exception as exc:
        logger.warning("VLM 食材结果解析失败: %s: %s", type(exc).__name__, exc)
        return None


def _vlm_vision(path: str, system: str, prompt: str) -> str | None:
    """统一的「图片 + 文本」多模态调用，失败返回 None。"""
    from app.adapters.llm import llm

    if not llm.available:
        return None

    try:
        img_b64 = _image_to_base64(path)
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_b64}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        resp = llm._client.chat.completions.create(
            model=settings.vlm_ingredient_model,
            messages=messages,
            temperature=0.1,
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception as exc:
        logger.warning("VLM 调用失败: %s: %s", type(exc).__name__, exc)
        return None


def _image_to_base64(path: str) -> str:
    """把图片转成 data URL（多模态 API 的标准传参格式）。"""
    suffix = Path(path).suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/jpeg")
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"
