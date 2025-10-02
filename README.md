# TensorRT FP16 сервис (Windows + Docker Desktop + WSL2)

## 1) Подготовка

- Установите NVIDIA драйвер на Windows (версия совместима с CUDA 12.1).

- Установите Docker Desktop с WSL2 backend и включите GPU support (Settings → Resources → WSL Integration + Use GPU).

- В WSL (Ubuntu) убедитесь, что `nvidia-smi` работает внутри `docker run --gpus all nvidia/cuda:12.1.0-base nvidia-smi`.

## 2) Экспорт в ONNX и сборка TensorRT FP16

В контейнере builder (у вас уже есть `docker/Dockerfile.builder`) выполните экспорт и сборку:

```bash
docker build -t trt-builder -f docker/Dockerfile.builder .
docker run --rm -ti --gpus all -v $(pwd):/workspace trt-builder

#в контейнере
pip install onnxscript
# Экспорт .pt -> .onnx (opset>=17, dynamic)
python app/models/export_onnx.py \
  --seg_pt app/models/segmentation_model.pt \
  --tip_pt app/models/tip_detector_model.pt \
  --seg_onnx app/models/segmentation_model.onnx \
  --tip_onnx app/models/tip_detector_model.onnx \
  

# Сборка .onnx -> .engine (FP16) для RTX 5080 sm_120
python app/models/build_trt.py \
 --seg_onnx models/segmentation_model.onnx \
 --tip_onnx models/tip_detector_model.onnx \
 --seg_engine models/segmentation_model_dynamic_fp16.engine \
 --tip_engine models/tip_detector_model_dynamic_fp16.engine \
 --precision fp16 \
 --workspace_mb 8192 \
 --min 1x3x960x1280 \
 --opt 1x3x1920x2560 \
 --max 1x3x2880x3840 \
 --hw_compat ampere+
```

  
## 3) Развернуть сервис на URL

```bash
docker compose -f compose_host.yml build
docker compose -f compose_host.yml up -d
```


## 4) Переключение backend

- В `.env` укажите:

```
BACKEND=torch      # для PyTorch
# или
BACKEND=tensorrt   # для TensorRT FP16
```

Перезапустите контейнер: `docker compose up -d`.

## 5) Примечания совместимости

- База образа: PyTorch 2.4.0 CUDA 12.1 runtime.
- Устанавливаются TensorRT runtime библиотеки (`libnvinfer*`).
- Для экспорта и сборки используйте builder образ, уже присутствующий в репозитории.