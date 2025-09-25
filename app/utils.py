import cv2
import numpy as np
from PIL import Image
import io
from pathlib import Path
from .model import model

STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

def detect_objects(image: Image.Image):
    # Конвертируем PIL → OpenCV (RGB → BGR)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Инференс через Ultralytics YOLO
    results = model(img)

    # Обрабатываем результаты
    detections = []
    result = results[0]  # берем первый (и единственный) результат

    # Если есть bounding boxes
    if result.boxes is not None:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            label = model.names[cls]

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [float(x) for x in xyxy]
            })

    # Рендерим изображение с bounding boxes
    annotated_img = result.plot()  # возвращает BGR numpy array
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    rendered_pil = Image.fromarray(annotated_img)

    return detections, rendered_pil

def save_image(image: Image.Image, filename: str) -> str:
    path = STATIC_DIR / filename
    image.save(path)
    return f"/static/{filename}"