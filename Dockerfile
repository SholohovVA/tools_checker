FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    UVICORN_WORKERS=1 \
    LOG_LEVEL=info


RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
        torch torchvision torchaudio

RUN pip install -i https://pypi.org/simple --default-timeout=120 \
        fastapi uvicorn[standard] python-multipart jinja2 \
        pillow opencv-python-headless ultralytics


RUN mkdir -p /app/static/original /app/static/results

COPY app ./app

RUN touch app/__init__.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
  CMD curl -fsS http://localhost:${APP_PORT}/healthz || curl -fsS http://localhost:${APP_PORT}/ || exit 1

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["sh", "-c", "uvicorn app.main:app --host ${APP_HOST} --port ${APP_PORT} --workers ${UVICORN_WORKERS} --log-level ${LOG_LEVEL}"]
