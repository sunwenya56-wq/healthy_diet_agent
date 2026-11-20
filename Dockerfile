# ---------- 基础镜像 ----------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ---------- 系统依赖 ----------
# ffmpeg 用于 webm→wav 转码（ASR 兜底）
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# ---------- Python 依赖 ----------
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------- 应用代码 ----------
COPY app/ ./app/
COPY data/ ./data/
COPY models/ ./models/

# 创建 upload 目录（运行时写入）
RUN mkdir -p /app/data/uploads/reports /app/data/uploads/fridge /app/data/uploads/voices \
 && chmod -R 755 /app/data

# ---------- 非 root 用户 ----------
RUN groupadd --gid 1000 appuser \
 && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# ---------- 健康检查 ----------
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

# ---------- 入口 ----------
# 容器内用 MySQL + Milvus（通过 compose 注入 URL），默认 SQLite + 内存向量
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
