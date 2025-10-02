import pathlib
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from app.model import seg_model, tip_model

CONFIDENCE = 0.0

# Словарь переименования классов
CLASS_MAPPING = [
    "Отвертка -",
    "Отвертка +",
    "Отвертка на смещенный крест",
    "Коловорот",
    "Пассатижи контровочные",
    "Пассатижи",
    "Шэрница",
    "Разводной ключ",
    "Открывашка для банок с маслом",
    "Ключ рожковый/накидной 3/4",
    "Бокорезы"
]

# Соответствие классов сегментации и детекции кончиков
TIPS_TO_SEG_CLASSES = {
    0: 3,
    1: 6,
    2: 7,
    3: 8,
    4: 9,
    5: 10,
    6: 0,
    7: 1,
    8: 2,
    9: 4,
    10: 5
}

STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)
pathlib.Path(STATIC_DIR / 'results').mkdir(parents=True, exist_ok=True)
pathlib.Path(STATIC_DIR / 'original').mkdir(parents=True, exist_ok=True)

print(f"Static dir exists: {STATIC_DIR.exists()}")
print(f"Original dir exists: {(STATIC_DIR / 'original').exists()}")
print(f"Results dir exists: {(STATIC_DIR / 'results').exists()}")


def detect_objects(image: Image.Image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    seg_results = seg_model.predict(image)[0]
    tip_results = tip_model.predict(image, conf=0.5)[0]

    # fix classes of boxes
    tip_results = tip_results.to('cpu')
    for i, cls in enumerate(tip_results.boxes.cls):
        tip_results.boxes.cls[i] = TIPS_TO_SEG_CLASSES[int(cls)]

    seg_boxes = extract_boxes(seg_results)
    tip_boxes = extract_boxes(tip_results)
    seg_items = extract_segmentations_with_masks(seg_results)

    # Запуск логики постобработки
    boxes, polygons = merge_segmentations_with_tips(seg_boxes, tip_boxes, seg_items)

    tip_items = extract_boxes_with_conf(tip_results)

    # Визуализация
    rendered_image = visualize_final_boxes(img, boxes, tip_items)

    # Сохраняем segmentation_data - НЕ ФИЛЬТРУЕМ здесь!
    segmentation_data = defaultdict(dict)
    for i, (box, poly_list) in enumerate(zip(boxes, polygons)):
        cls_id = int(box[-1])
        conf = box[-2]

        segmentation_data[cls_id] = {
            "confidence": conf,
            "bbox": box[:4],
            "polygons": poly_list  # Сохраняем все полигоны
        }

        print(f"DEBUG: Final segmentation_data for class {cls_id}: {len(poly_list)} polygons")

    return boxes, rendered_image, segmentation_data


def save_image(image, filename: str, is_original=False) -> str:
    """
    Сохраняет изображение в соответствующую директорию используя PIL
    """
    if is_original:
        path = STATIC_DIR / 'original' / filename
        # Если image это numpy array, конвертируем в PIL Image
        if isinstance(image, np.ndarray):
            # Если image в формате BGR (от cv2), конвертируем в RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image)
        else:
            image_pil = image

        image_pil.save(str(path), format='JPEG', quality=95)
        print(f"Original image saved: {path}")
        return f"/static/original/{filename}"
    else:
        path = STATIC_DIR / 'results' / filename
        # Для обработанных изображений
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image)
        else:
            image_pil = image

        image_pil.save(str(path), format='JPEG', quality=95)
        print(f"Processed image saved: {path}")
        return f"/static/results/{filename}"


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

    # Запуск логики постобработки
    detections = merge_segmentations_with_tips(seg_boxes, tip_boxes)

    return detections, w, h


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
        tip_boxes: List[Tuple[int, float, List[float]]],
        seg_items: List[Tuple[int, float, List[float], List]] = None
):
    print(f"DEBUG: Input seg_boxes: {len(seg_boxes)}, seg_items: {len(seg_items) if seg_items else 0}")

    # Детальная отладка seg_items
    if seg_items:
        for i, (cls_id, conf, box, polygon) in enumerate(seg_items):
            print(
                f"DEBUG: seg_item {i}: class={cls_id}, conf={conf:.2f}, bbox={box[:4]}, polygon_points={len(polygon)}")
            if len(polygon) < 3:
                print(f"DEBUG: WARNING: seg_item {i} has only {len(polygon)} points!")

    # Группируем сегментации по классам
    seg_by_class = {}
    for cls_id, conf, box in seg_boxes:
        seg_by_class.setdefault(cls_id, []).append((conf, box))

    # Группируем кончики по классам (только координаты)
    tips_by_class = {}
    for cls_id, _, box in tip_boxes:
        tips_by_class.setdefault(cls_id, []).append(box)

    # Группируем полигоны по классам - ВАЖНО: не фильтруем здесь!
    polygons_by_class = defaultdict(list)
    if seg_items:
        for cls_id, conf, box, polygon in seg_items:
            # НЕ ФИЛЬТРУЕМ здесь - сохраняем все полигоны
            polygons_by_class[cls_id].append(polygon)
            print(f"DEBUG: Added polygon for class {cls_id}: {len(polygon)} points")

    output_boxes = []
    output_polygons = []

    for cls_id, seg_list in seg_by_class.items():
        class_polygons = polygons_by_class.get(cls_id, [])
        print(f"DEBUG: Class {cls_id} has {len(seg_list)} segments and {len(class_polygons)} polygons")

        # Отладочная информация о полигонах этого класса
        for i, poly in enumerate(class_polygons):
            print(f"DEBUG:   Class {cls_id} polygon {i}: {len(poly)} points")

        if len(seg_list) == 1:
            # Один сегмент → один объект
            conf, box = seg_list[0]
            output_boxes.append([*box, conf, cls_id])
            if class_polygons:
                output_polygons.append(class_polygons)
            else:
                output_polygons.append([])
        else:
            # ... остальная логика без изменений ...
            boxes_only = [b for _, b in seg_list]
            x1s = [b[0] for b in boxes_only]
            y1s = [b[1] for b in boxes_only]
            x2s = [b[2] for b in boxes_only]
            y2s = [b[3] for b in boxes_only]
            merged_box = [min(x1s), min(y1s), max(x2s), max(y2s)]

            tips_in_box = 0
            if cls_id in tips_by_class:
                for tip_box in tips_by_class[cls_id]:
                    tip_cx = (tip_box[0] + tip_box[2]) / 2
                    tip_cy = (tip_box[1] + tip_box[3]) / 2
                    if (merged_box[0] <= tip_cx <= merged_box[2] and
                            merged_box[1] <= tip_cy <= merged_box[3]):
                        tips_in_box += 1

            if tips_in_box <= 1:
                avg_conf = sum(conf for conf, _ in seg_list) / len(seg_list)
                output_boxes.append([*merged_box, avg_conf, cls_id])
                # Объединяем все полигоны класса
                output_polygons.append(class_polygons)
            else:
                for i, (conf, box) in enumerate(seg_list):
                    output_boxes.append([*box, conf, cls_id])
                    # Сохраняем соответствующий полигон
                    polygon = []
                    if i < len(class_polygons):
                        polygon = [class_polygons[i]]  # Сохраняем как список с одним полигоном
                    output_polygons.append(polygon)

    # Финальная отладка
    print(f"DEBUG: Output - {len(output_boxes)} boxes, {len(output_polygons)} polygon lists")
    for i, (box, poly_list) in enumerate(zip(output_boxes, output_polygons)):
        print(f"DEBUG: Output {i}: class={int(box[-1])}, polygons={len(poly_list)}")
        for j, poly in enumerate(poly_list):
            print(f"DEBUG:   Polygon {j}: {len(poly)} points")

    return output_boxes, output_polygons


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
    (0, 128, 128),  # combination_pliers
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
        final_boxes: List[List[float]],
        tip_boxes: List[List[float]]
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
        label = f"{CLASS_MAPPING[cls_id]} {conf:.2f}"

        # Рисуем бокс
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 5)

        # Фон для подписи
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.6, 2)
        cv2.rectangle(vis_img, (x1, y1 - 25), (x1 + w, y1), color, -1)

        # Подпись
        cv2.putText(
            vis_img,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_COMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    for cls_id, conf, box in tip_boxes:
        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_MAPPING[cls_id]} {conf:.2f}"

        # Фон для подписи
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.3, 2)
        cv2.rectangle(vis_img, (x1, y1 - 25), (x1 + w, y1), color, -1)

        # Подпись
        cv2.putText(
            vis_img,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_COMPLEX,
            0.3,
            (255, 255, 255),
            1
        )

    return vis_img


def extract_segmentations_with_masks(results):
    items = []
    if results.masks is None or len(results.masks) == 0:
        return items
    boxes = results.boxes
    masks = results.masks
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        box = boxes.xyxy[i].cpu().numpy().tolist()
        # Полигон: (N, 2) -> список кортежей
        polygon = masks.xy[i].astype(int).tolist()  # уже в пикселях
        items.append((cls_id, conf, box, polygon))
    return items


def extract_boxes_with_conf(results):
    items = []
    if results.boxes is None or len(results.boxes) == 0:
        return items
    for i in range(len(results.boxes)):
        cls_id = int(results.boxes.cls[i].item())
        conf = float(results.boxes.conf[i].item())
        box = results.boxes.xyxy[i].cpu().numpy().tolist()
        items.append((cls_id, conf, box))
    return items


def visualize_debug_with_polygons_and_tip_boxes(
        image: np.ndarray,
        seg_items: List[Tuple[int, float, List[float], List[Tuple[int, int]]]],
        tip_items: List[Tuple[int, float, List[float]]]
) -> np.ndarray:
    """
    Визуализация:
    - Сегментации: как полигоны (заливка + контур)
    - Кончики: как bounding boxes (прямоугольники)
    Args:
        image: исходное изображение (BGR)
        seg_items: [(cls_id, conf, box, polygon), ...]
        tip_items: [(cls_id, conf, box), ...]
    Returns:
        np.ndarray: аннотированное изображение (BGR)
    """
    vis_img = image.copy()
    overlay = vis_img.copy()
    for cls_id, conf, box, polygon in seg_items:
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        # Заливка полупрозрачная
        cv2.fillPoly(overlay, [np.array(polygon, dtype=np.int32)], color)
        # Контур
        cv2.polylines(vis_img, [np.array(polygon, dtype=np.int32)], isClosed=True, color=color, thickness=2)
        # Подпись у bounding box (для позиционирования)
        x1, y1, x2, y2 = map(int, box)
        label = f"S:{CLASS_NAMES[cls_id][:4]} {conf:.2f}"
        cv2.putText(vis_img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    # Наложение полупрозрачной заливки
    alpha = 0.3
    vis_img = cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0)
    # 2. Рисуем bounding boxes кончиков
    for cls_id, conf, box in tip_items:
        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        label = f"T:{CLASS_NAMES[cls_id][:4]} {conf:.2f}"
        cv2.putText(vis_img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis_img
