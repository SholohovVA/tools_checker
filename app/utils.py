import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from .model import model

CONFIDENCE = 0.0

# Словарь переименования классов
# CLASS_MAPPING = {
#     "minus_screwdriver": "Отвертка «-»",
#     "plus_screwdriver": "Отвертка «+»",
#     "offset_phillips_screwdriver": "Отвертка на смещенный крест",
#     "brace": "Коловорот",
#     "locking_pliers": "Пассатижи контровочные",
#     "combination_pliers": "Пассатижи",
#     "shernica": "Шэрница",
#     "adjustable_wrench": "Разводной ключ",
#     "oil_can_opener": "Открывашка для банок с маслом",
#     "open_end_wrench": "Ключ рожковый/накидной ¾",
#     "side_cutting_pliers": "Бокорезы"
# }

#Словарь переименования классов
CLASS_MAPPING = {
    "screwdriver": "Отвертка",
    "pliers": "Пассатижи",
    "brace": "Коловорот",
    "shernica": "Шэрница",
    "adjustable_wrench": "Разводной ключ",
    "oil_can_opener": "Открывашка для банок с маслом",
    "open-end_wrench": "Ключ рожковый/накидной ¾",
    "side_cutting_pliers": "Бокорезы"
}

STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

def detect_objects(image: Image.Image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    #results = model(img, conf=CONFIDENCE)
    results = model.predict(img, conf=0.25, iou = 0.2, imgsz=900)
    detections = []
    result = results[0]

    if result.boxes is not None:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            original_label = model.names[cls]
            # Применяем маппинг, если есть; иначе оставляем оригинал
            display_label = CLASS_MAPPING.get(original_label, original_label)

            detections.append({
                "original_label": original_label,
                "label": display_label,
                "confidence": conf,
                "bbox": [float(x) for x in xyxy]
            })

    annotated_img = result.plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    rendered_pil = Image.fromarray(annotated_img)

    return detections, rendered_pil

def save_image(image: Image.Image, filename: str) -> str:
    path = STATIC_DIR / filename
    image.save(path)
    return f"/static/{filename}"

def bbox_to_yolo_format(xyxy, img_width, img_height):
    """Преобразует [x1, y1, x2, y2] → [x_center, y_center, w, h] (нормализовано)"""
    x1, y1, x2, y2 = xyxy
    dw = 1.0 / img_width
    dh = 1.0 / img_height
    x = (x1 + x2) / 2.0
    y = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [round(x, 6), round(y, 6), round(w, 6), round(h, 6)]

def detect_objects_with_meta(image: Image.Image):
    img = np.array(image)
    h, w = img.shape[:2]
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = model(img_bgr)
    detections = []
    result = results[0]

    if result.boxes is not None:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            original_label = model.names[cls_id]
            display_label = CLASS_MAPPING.get(original_label, original_label)

            yolo_bbox = bbox_to_yolo_format(xyxy, w, h)

            detections.append({
                "class_id": cls_id,
                "original_label": original_label,
                "label": display_label,
                "confidence": conf,
                "bbox_xyxy": xyxy.tolist(),
                "bbox_yolo": yolo_bbox
            })

    return detections, w, h

def convert_to_serializable(obj):
    """Рекурсивно преобразует numpy-типы в стандартные Python-типы."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj