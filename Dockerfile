
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

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
COPY models ./models

RUN touch app/__init__.py

EXPOSE 8000

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
