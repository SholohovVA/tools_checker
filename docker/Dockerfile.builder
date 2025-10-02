FROM nvcr.io/nvidia/tensorrt:25.09-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# PyTorch с поддержкой SM 12.0 (Blackwell)
RUN pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# ONNX / ORT GPU (по желанию можно на CPU: onnxruntime)
RUN pip install onnx onnxruntime-gpu

# NumPy<2 для совместимости со сторонними либами
RUN pip install "numpy<2.0" opencv-python-headless pillow scipy

# Если используешь YOLO/Ultralytics
RUN pip install ultralytics

# Утилиты для TRT/ONNX
RUN pip install polygraphy onnx-graphsurgeon

WORKDIR /workspace
COPY app/ ./app/
COPY models/ ./models/

RUN mkdir -p /workspace/app/models/dataset_calib \
 && mkdir -p /workspace/static/original /workspace/static/results

ENV PYTHONPATH=/workspace
ENTRYPOINT ["bash"]