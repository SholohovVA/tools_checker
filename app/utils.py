import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from model import seg_model, tip_model
from typing import List, Tuple, Dict

CONFIDENCE = 0.0

# Словарь переименования классов
CLASS_MAPPING = [
    "Отвертка «-»",
    "Отвертка «+»",
    "Отвертка на смещенный крест",
    "Коловорот",
    "Пассатижи контровочные",
    "Пассатижи",
    "Шэрница",
    "Разводной ключ",
    "Открывашка для банок с маслом",
    "Ключ рожковый/накидной ¾",
    "Бокорезы"
]

STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

def detect_objects(image: Image.Image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    seg_results = seg_model(image)[0]
    tip_results = tip_model(image)[0]

    seg_boxes = extract_boxes(seg_results)
    tip_boxes = extract_boxes(tip_results)

    # # Группируем сегментации по классам (нужно для визуализации и логики)
    # seg_by_class = {}
    # for cls_id, box in seg_boxes:
    #     seg_by_class.setdefault(cls_id, []).append(box)

    # Запуск логики постобработки
    boxes = merge_segmentations_with_tips(seg_boxes, tip_boxes)

    # Визуализация
    rendered_image = visualize_final_boxes(img, boxes)

    return boxes, rendered_image

def save_image(image, filename: str) -> str:
    path = STATIC_DIR / filename
    cv2.imwrite(path, image)
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

    seg_results = seg_model(image)[0]
    tip_results = tip_model(image)[0]

    seg_boxes = extract_boxes(seg_results)
    tip_boxes = extract_boxes(tip_results)

    # # Группируем сегментации по классам (нужно для визуализации и логики)
    # seg_by_class = {}
    # for cls_id, box in seg_boxes:
    #     seg_by_class.setdefault(cls_id, []).append(box)

    # Запуск логики постобработки
    boxes = merge_segmentations_with_tips(seg_boxes, tip_boxes)

    return boxes, w, h


def extract_boxes(results):
    boxes = []
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            boxes.append((cls_id, conf, [x1, y1, x2, y2]))
    return boxes


def merge_segmentations_with_tips(
        seg_boxes: List[Tuple[int, float, List[float]]],
        tip_boxes: List[Tuple[int, float, List[float]]]
) -> List[List[float]]:
    """
    Постобработка: объединение сегментаций с использованием детекции кончиков.
    Возвращает итоговые bounding boxes в формате YOLO: [x1, y1, x2, y2, conf, cls]

    Args:
        seg_boxes: [(class_id, conf, [x1, y1, x2, y2]), ...]
        tip_boxes: [(class_id, conf, [x1, y1, x2, y2]), ...]

    Returns:
        List of [x1, y1, x2, y2, conf, cls]
    """
    # Группируем сегментации по классам
    seg_by_class = {}
    for cls_id, conf, box in seg_boxes:
        seg_by_class.setdefault(cls_id, []).append((conf, box))

    # Группируем кончики по классам (только координаты)
    tips_by_class = {}
    for cls_id, _, box in tip_boxes:
        tips_by_class.setdefault(cls_id, []).append(box)

    output_boxes = []

    for cls_id, seg_list in seg_by_class.items():
        if len(seg_list) == 1:
            # Один сегмент → один объект
            conf, box = seg_list[0]
            output_boxes.append([*box, conf, cls_id])
        else:
            # Объединяем все боксы для проверки зоны
            boxes_only = [b for _, b in seg_list]
            x1s = [b[0] for b in boxes_only]
            y1s = [b[1] for b in boxes_only]
            x2s = [b[2] for b in boxes_only]
            y2s = [b[3] for b in boxes_only]
            merged_box = [min(x1s), min(y1s), max(x2s), max(y2s)]

            # Считаем кончики этого класса внутри merged_box
            tips_in_box = 0
            if cls_id in tips_by_class:
                for tip_box in tips_by_class[cls_id]:
                    tip_cx = (tip_box[0] + tip_box[2]) / 2
                    tip_cy = (tip_box[1] + tip_box[3]) / 2
                    if (merged_box[0] <= tip_cx <= merged_box[2] and
                            merged_box[1] <= tip_cy <= merged_box[3]):
                        tips_in_box += 1

            if tips_in_box == 0:
                # Нет кончиков, отбрасываем все сегментации
                avg_conf = sum(conf for conf, _ in seg_list) / len(seg_list)
                output_boxes.append([*merged_box, avg_conf, cls_id])
            elif tips_in_box == 1:
                # Один кончик, один объект: мержим
                avg_conf = sum(conf for conf, _ in seg_list) / len(seg_list)
                output_boxes.append([*merged_box, avg_conf, cls_id])
            else:  # tips_in_box >= 2
                # Много кончиков → разные объекты: возвращаем все исходные сегментации
                for conf, box in seg_list:
                    output_boxes.append([*box, conf, cls_id])

    return output_boxes

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


# Цвета и имена классов (те же, что и раньше)
CLASS_COLORS = [
    (255, 0, 0),  # minus_screwdriver
    (0, 255, 0),  # plus_screwdriver
    (0, 0, 255),  # offset_phillips_screwdriver
    (255, 255, 0),  # brace
    (255, 0, 255),  # locking_pliers
    (0, 255, 255),  # combination_pliers
    (128, 0, 0),  # shernica
    (0, 128, 0),  # adjustable_wrench
    (0, 0, 128),  # oil_can_opener
    (128, 128, 0),  # open_end_wrench
    (128, 0, 128),  # side_cutting_pliers
]

CLASS_NAMES = [
    "minus_screwdriver",
    "plus_screwdriver",
    "offset_phillips_screwdriver",
    "brace",
    "locking_pliers",
    "combination_pliers",
    "shernica",
    "adjustable_wrench",
    "oil_can_opener",
    "open_end_wrench",
    "side_cutting_pliers"
]


def visualize_final_boxes(
        image: np.ndarray,
        final_boxes: List[List[float]]
) -> np.ndarray:
    """
    Визуализация итоговых bounding boxes после постобработки.

    Args:
        image (np.ndarray): исходное изображение в формате OpenCV (BGR, HxWx3)
        final_boxes (List[List[float]]): список боксов в формате [x1, y1, x2, y2, conf, cls]

    Returns:
        np.ndarray: изображение с нарисованными боксами (BGR)
    """
    vis_img = image.copy()

    for box in final_boxes:
        x1, y1, x2, y2, conf, cls_id = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls_id = int(cls_id)

        # Получаем цвет и имя
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"

        # Рисуем бокс
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

        # Фон для подписи
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis_img, (x1, y1 - 25), (x1 + w, y1), color, -1)

        # Подпись
        cv2.putText(
            vis_img,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    return vis_img
