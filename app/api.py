"""FastAPI 服务层：Agent 运行、流式输出、上传、菜谱查询。"""

import base64
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.adapters.vision import ReportParseError
from app.config import BASE_DIR, settings
from app.db import ensure_user, init_db, list_recipes, save_recipe
from app.graph import get_graph, run_agent
from app.streaming import iter_sse_lines
from app.tools.implementations import build_registry, tool_detect_ingredients, tool_parse_report

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY = build_registry()
logger = logging.getLogger(__name__)


class RunRequest(BaseModel):
    user_id: int = 1
    query: str = Field(default="根据我的体检报告和冰箱食材，推荐一份晚餐")
    session_id: str | None = None
    report_image_path: str | None = None
    fridge_image_path: str | None = None
    ingredients: list[str] | None = None
    debug: bool = False


def _public(result: dict[str, Any], debug: bool) -> dict[str, Any]:
    payload = {
        "session_id": result.get("session_id"),
        "intent": result.get("intent"),
        "profile": result.get("profile", {}),
        "health_metrics": result.get("health_metrics", {}),
        "ingredients": result.get("ingredients", []),
        "knowledge_hits": result.get("retrieved_knowledge", []),
        "nutrition": result.get("nutrition_report", {}),
        "plan": result.get("plan", []),
        "reflection": result.get("reflection", {}),
        "revision_count": result.get("revision_count", 0),
        "final_answer": result.get("final_answer", ""),
        "errors": result.get("errors", []),
    }
    if debug:
        payload["tool_trace"] = result.get("tool_trace", [])
    return payload


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="Healthy Diet Agent", version="2.0.0", lifespan=lifespan)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "llm_mock": settings.llm_mock,
        "vision_mock": settings.vision_mock,
        "vector_backend": settings.vector_backend,
        "database": settings.database_url.split("://")[0],
    }


@app.get("/api/tools")
def list_tools() -> dict[str, Any]:
    """MCP tools/list 的等价接口，前端/调试可直接看到模型能调用什么。"""
    return {"tools": REGISTRY.schemas()}


@app.post("/api/agent/run")
def agent_run(req: RunRequest) -> dict[str, Any]:
    ensure_user(req.user_id)
    result = run_agent(
        query=req.query,
        user_id=req.user_id,
        session_id=req.session_id,
        report_image_path=req.report_image_path,
        fridge_image_path=req.fridge_image_path,
        ingredients=req.ingredients,
        graph=get_graph(),
    )
    recipe_id = save_recipe(req.user_id, result)
    payload = _public(result, req.debug)
    payload["recipe_id"] = recipe_id
    return payload


@app.post("/api/agent/stream")
def agent_stream(req: RunRequest) -> StreamingResponse:
    ensure_user(req.user_id)

    def _gen():
        yield from iter_sse_lines(
            query=req.query,
            user_id=req.user_id,
            session_id=req.session_id,
            report_image_path=req.report_image_path,
            fridge_image_path=req.fridge_image_path,
            ingredients=req.ingredients,
            graph=get_graph(),
        )

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/upload/fridge")
def upload_fridge(user_id: int = Form(default=1), file: UploadFile = File(...)) -> dict[str, Any]:
    path = _save_upload(file, "fridge")
    out = tool_detect_ingredients({"image_path": str(path)})
    ensure_user(user_id)
    return {"path": str(path), "ingredients": out["ingredients"], "raw": out["raw"]}


@app.post("/api/upload/report")
def upload_report(user_id: int = Form(default=1), file: UploadFile = File(...)) -> dict[str, Any]:
    path = _save_upload(file, "reports")
    try:
        out = tool_parse_report({"image_path": str(path)})
    except ReportParseError as exc:
        ensure_user(user_id)
        return {"path": str(path), "text": "", "source": "error", "error": str(exc)}
    ensure_user(user_id)
    return {"path": str(path), "text": out["text"], "source": out["source"]}


@app.get("/api/recipes")
def recipes(user_id: int = 1, limit: int = 20) -> dict[str, Any]:
    return {"items": list_recipes(user_id, limit)}


@app.post("/api/asr")
def asr(user_id: int = Form(default=1), file: UploadFile = File(...)) -> dict[str, Any]:
    """语音转写：硅基流动 XingChenGSR（真实）→ Mock 兜底。

    配置 ASR_API_KEY（或 SILICONFLOW_API_KEY）后走真实转写；
    未配置或调用失败才回退样例文本，并明确返回 source="mock"。
    """
    path = _save_upload(file, "voices")
    ensure_user(user_id)

    text = _siliconflow_asr(path)
    if text:
        return {"text": text, "source": settings.asr_model, "path": str(path)}

    # 已配 key 说明是真实调用失败：不编造文本，交给前端提示
    if settings.asr_api_key:
        return {"text": "", "source": "error", "path": str(path), "note": "ASR 调用失败或未识别到语音"}

    # 未配 key：纯 Demo 样例，按文件大小稳定取样例
    samples = [
        "根据我的体检报告和冰箱里的食材推荐一份晚餐",
        "我有高血压，给我做一份清淡的午餐",
        "用西红柿鸡蛋和青椒做个快手菜",
        "推荐一份适合减脂的早餐搭配",
        "我血糖有点高，帮我规划一周的菜谱",
    ]
    idx = int(path.stat().st_size) % len(samples)
    return {"text": samples[idx], "source": "mock", "path": str(path), "note": "未配置 ASR key，返回样例文本"}


class _TTsRequest(BaseModel):
    text: str


@app.post("/api/tts")
def tts(req: _TTsRequest) -> dict[str, Any]:
    """文本转语音：Windows 自带 SAPI（离线、免费）→ Mock 静音兜底。"""
    text = (req.text or "").strip()

    audio_base64 = _windows_tts(text)
    if audio_base64:
        return {"audio_base64": audio_base64, "mime": "audio/wav", "source": "system-speech"}

    return {
        "audio_base64": _mock_silent_wav(),
        "mime": "audio/wav",
        "source": "mock",
        "note": "系统语音合成不可用，返回静音占位",
    }


def _mock_silent_wav() -> str:
    """最小 8kHz 单声道 1 秒静音 WAV 的 base64。"""
    import base64
    import struct

    sample_rate = 8000
    duration = 1
    n_channels = 1
    n_samples = sample_rate * duration
    byte_rate = sample_rate * n_channels * 1
    block_align = n_channels
    bits_per_sample = 8

    header = (
        b"RIFF"
        + struct.pack("<I", 36 + n_samples)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate, byte_rate, block_align, bits_per_sample)
        + b"data"
        + struct.pack("<I", n_samples)
    )
    return base64.b64encode(header + b"\x80" * n_samples).decode("ascii")


def _siliconflow_asr(path: Path) -> str | None:
    """硅基流动语音转写（OpenAI 兼容 /audio/transcriptions）。

    浏览器 MediaRecorder 产出的是 webm，部分服务端不接受该容器，
    失败时用 ffmpeg 转成 16kHz 单声道 wav 再试一次。
    """
    if not settings.asr_api_key:
        logger.info("ASR 未配置 key，跳过真实转写")
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.asr_api_key, base_url=settings.asr_base_url, timeout=60.0)
        try:
            return _transcribe(client, path)
        except Exception as exc:
            if path.suffix.lower() in {".wav", ".mp3"}:
                raise
            logger.warning("ASR 首次调用失败（%s），尝试转码为 wav 重试", type(exc).__name__)

        wav = _to_wav(path)
        if not wav:
            return None
        try:
            return _transcribe(client, wav)
        finally:
            wav.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("ASR 调用失败: %s: %s", type(exc).__name__, exc)
        return None


def _transcribe(client: Any, path: Path) -> str | None:
    with path.open("rb") as audio:
        resp = client.audio.transcriptions.create(model=settings.asr_model, file=audio)
    return (resp.text or "").strip() or None


def _to_wav(path: Path) -> Path | None:
    """用 ffmpeg 转码成 16kHz 单声道 wav；没有 ffmpeg 时返回 None。"""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    target = path.with_suffix(".asr.wav")
    result = subprocess.run(
        [ffmpeg, "-y", "-i", str(path), "-ar", "16000", "-ac", "1", str(target)],
        capture_output=True,
    )
    if result.returncode != 0 or not target.exists():
        return None
    return target


def _windows_tts(text: str) -> str | None:
    """用 Windows 自带 SAPI 离线合成 wav，无需第三方包。

    长文本按中文标点切成 ~200 字/段，分段合成后同采样率拼接，
    避免单次 Speak 对超长文本截断（Windows SAPI 单次上限约数千字）。
    优先选中中文发音人（如 Microsoft Huihui）。
    """
    if not text:
        return None

    segments = _split_for_tts(text)
    if not segments:
        return None

    import io
    import tempfile
    import wave

    chunks: list[bytes] = []
    wav_params: wave._WaveParams | None = None

    for seg in segments:
        wav_bytes = _synthesize_one_segment(seg)
        if not wav_bytes:
            continue
        chunks.append(wav_bytes)
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            if wav_params is None:
                wav_params = w.getparams()

    if not chunks or wav_params is None:
        return None

    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setparams(wav_params)
        for chunk in chunks:
            with wave.open(io.BytesIO(chunk), "rb") as w:
                out.writeframes(w.readframes(w.getnframes()))

    full = buf.getvalue()
    return base64.b64encode(full).decode("ascii") if len(full) > 44 else None


_TTS_SENT_RE = re.compile(r"(?<=[。！？!?；;])\s*|\n+")
_TTS_MAX_PER_SEG = 200


def _split_for_tts(text: str, max_len: int = _TTS_MAX_PER_SEG) -> list[str]:
    """按中文标点切片，每段不超过 max_len 字。"""
    out: list[str] = []
    cur = ""
    for seg in _TTS_SENT_RE.split(text):
        piece = seg.strip()
        if not piece:
            continue
        if len(cur) + len(piece) <= max_len:
            cur += piece
        else:
            if cur:
                out.append(cur)
            if len(piece) <= max_len:
                cur = piece
            else:
                # 无标点可切就硬切
                for i in range(0, len(piece), max_len):
                    out.append(piece[i : i + max_len])
                cur = ""
    if cur:
        out.append(cur)
    return out


def _synthesize_one_segment(text: str) -> bytes | None:
    """单段合成：写临时文件 → 子进程调用 System.Speech → 读回 wav。"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        text_file = Path(tmp) / "text.txt"
        out_file = Path(tmp) / "tts.wav"
        text_file.write_text(text, encoding="utf-8")

        script = (
            "Add-Type -AssemblyName System.Speech;"
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            "$zh = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'zh*' } | Select-Object -First 1;"
            "if ($zh) { $s.SelectVoice($zh.VoiceInfo.Name) };"
            f"$s.SetOutputToWaveFile('{out_file}');"
            "$s.Speak([IO.File]::ReadAllText($env:TTS_TEXT_FILE, [Text.Encoding]::UTF8));"
            "$s.Dispose()"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                check=True,
                capture_output=True,
                env={**os.environ, "TTS_TEXT_FILE": str(text_file)},
            )
        except Exception as exc:
            logger.warning("系统 TTS 单段合成失败: %s: %s", type(exc).__name__, exc)
            return None

        data = out_file.read_bytes() if out_file.exists() else b""
        return data if len(data) > 44 else None


def _save_upload(file: UploadFile, category: str) -> Path:
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".pdf", ".webp"}
    AUDIO_EXTS = {".wav", ".mp3", ".webm", ".m4a", ".ogg", ".flac"}
    ALLOWED = IMAGE_EXTS | AUDIO_EXTS

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {suffix}")
    target = UPLOAD_DIR / category
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{uuid.uuid4().hex}{suffix}"
    with path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return path
