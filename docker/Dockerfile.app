FROM nvcr.io/nvidia/tensorrt:25.09-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-opencv libgl1 libglib2.0-0 curl git ca-certificates build-essential\
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt || true
RUN pip install --no-cache-dir fastapi uvicorn[standard] pillow opencv-python-headless || true

RUN pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
        torch torchvision torchaudio

WORKDIR /app

COPY app /app/app
RUN mkdir -p /app/static

ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000
ENV UVICORN_WORKERS=1
ENV BACKEND=torch

EXPOSE 8000

CMD ["bash", "-lc", "exec python -m uvicorn app.main:app --host ${APP_HOST:-0.0.0.0} --port ${APP_PORT:-8000} --workers ${UVICORN_WORKERS:-1}"]



