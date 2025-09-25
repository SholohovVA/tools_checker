import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from .model import model

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

# Словарь переименования классов
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

    results = model(img)
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