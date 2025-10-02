import os
import uuid
from collections import defaultdict

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn

from app.utils import CLASS_MAPPING
from app.utils_verification import ImprovedSiameseNetwork, add_padding_to_square, preprocess_masks_on_image


# from train_identification_vgg16 import SiameseNetwork


def load_verification_model(model_name, weights_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    INPUT_SIZE = 320

    # Загрузка модели
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = ImprovedSiameseNetwork(embedding_dim=256, input_size=INPUT_SIZE, backbone=model_name).to(device)

    try:
        # checkpoint = torch.load('models/checkpoint_best_resnet_v1.pth', map_location=device)
        checkpoint = torch.load(f'models/{weights_name}', map_location=device)

        # Проверяем структуру checkpoint
        print("Keys in checkpoint:", checkpoint.keys())
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            print("First few keys in model_state_dict:", list(model_state_dict.keys())[:10])

            # Загружаем state dict с обработкой несовпадений
            model.load_state_dict(model_state_dict, strict=False)
            print("Model loaded successfully with strict=False")
        else:
            print("No 'model_state_dict' found in checkpoint")

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using randomly initialized model")

    model.eval()
    return model, transform


THRESHOLD = 0.55
# model_name = 'resnet50'
# weights_name = 'checkpoint_best_resnet_v1.pth'

model_name = 'mobilenet_v3_small'
weights_name = 'checkpoint_best_mobilenet_v3s_39ep.pth'
verification_model, transform = load_verification_model(model_name, weights_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def verify_images(img1, img2, threshold=0.7):
    """
    Верификация двух изображений с помощью Siamese сети
    """
    try:
        img1_tensor = transform(img1).unsqueeze(0).to(device)
        img2_tensor = transform(img2).unsqueeze(0).to(device)

        with torch.no_grad():
            emb1, emb2 = verification_model(img1_tensor, img2_tensor)
            similarity = nn.functional.cosine_similarity(emb1, emb2)

        is_same = similarity.item() > threshold
        return is_same, similarity.item()
    except Exception as e:
        print(f"Ошибка при попытке идентификации изображений: {e}")
        return False, 0.0


def verificate_objects(data_taken, data_returned):
    det_taken = data_taken["detections"]
    seg_taken = data_taken["segmentation_data"]
    url_taken = data_taken["original_url"]
    det_returned = data_returned["detections"]
    seg_returned = data_returned["segmentation_data"]
    url_returned = data_returned["original_url"]
    try:
        # img_taken = Image.open(url_taken.replace('/static/', 'static/')).convert('RGB')
        img_taken_src = cv2.imread(url_taken.replace('/static/', 'static/'))
        # img_returned = Image.open(url_returned.replace('/static/', 'static/')).convert('RGB')
        img_returned_src = cv2.imread(url_returned.replace('/static/', 'static/'))
    except Exception as e:
        print(f"Ошибка загрузки изображений: {e}")
        return []

    verification_results = defaultdict(dict)

    det_taken_by_class = defaultdict(list)
    for det_t in det_taken:
        # for data in det:
        if len(det_t) == 0:
            continue
        x1, y1, x2, y2, conf, cls_id = det_t
        # det_taken_by_class.setdefault(cls_id, []).append([x1, y1, x2, y2])
        det_taken_by_class[cls_id] = [x1, y1, x2, y2]

    det_returned_by_class = defaultdict(list)
    for det_r in det_returned:
        # for data in det_ret:
        if len(det_r) == 0:
            continue
        x1, y1, x2, y2, conf, cls_id = det_r
        det_returned_by_class[cls_id] = [x1, y1, x2, y2]

    # Проходим по всем детекциям в taken
    for cls_id_t, bbox_t in det_taken_by_class.items():

        seg_t = seg_taken[cls_id_t]
        polygons_count = len(seg_t.get("polygons", []))
        print(f"DEBUG: Verification - Class {cls_id_t} has {polygons_count} polygons")
        if polygons_count > 0:
            for i, poly in enumerate(seg_t["polygons"]):
                print(f"DEBUG:   Polygon {i}: {len(poly)} points")
        # обрезаем изображение
        crop_taken = preprocess_masks_on_image(img_taken_src, seg_t, bbox_t)
        crop_taken_resized = add_padding_to_square(crop_taken, target_size=320)
        # x1_t, y1_t, x2_t, y2_t = map(int, bbox_t)
        # crop_taken = img_taken.crop((x1_t, y1_t, x2_t, y2_t))

        if cls_id_t not in det_returned_by_class or cls_id_t not in seg_returned:
            verification_results[cls_id_t] = {
                "class_name": CLASS_MAPPING[cls_id_t],
                "similarity": 0.0,
                "is_same": False,
                "status": "class_missing",
                "message": f"Класс {cls_id_t} отсутствует в возвращенных инструментах"
            }
            continue

        seg_r = seg_returned[cls_id_t]
        bbox_r = det_returned_by_class[cls_id_t]

        crop_returned = preprocess_masks_on_image(img_returned_src, seg_r, bbox_r)
        crop_returned_resized = add_padding_to_square(crop_returned, target_size=320)

        # x1_r, y1_r, x2_r, y2_r = map(int, bbox_r)
        # crop_returned = img_returned.crop((x1_r, y1_r, x2_r, y2_r))

        # Верифицируем обрезанные области
        try:
            # Сохраняем временные файлы для верификации
            temp_taken_path = f"static/verification/taken_crop_{cls_id_t}.jpg"
            temp_returned_path = f"static/verification/returned_crop_{cls_id_t}.jpg"

            # Создаем временную директорию если нет
            os.makedirs("static/verification", exist_ok=True)

            crop_taken_resized.save(temp_taken_path)
            crop_returned_resized.save(temp_returned_path)

            # Выполняем верификацию
            is_same, similarity = verify_images(crop_taken_resized, crop_returned_resized, threshold=THRESHOLD)
            verification_results[cls_id_t] = {
                "class_name": CLASS_MAPPING[cls_id_t],
                "similarity": similarity,
                "is_same": is_same,
                "status": "verified"
            }

        except Exception as e:
            print(f"Ошибка при верификации класса {cls_id_t}: {e}")
            verification_results[cls_id_t] = {
                "class_name": CLASS_MAPPING[cls_id_t],
                "similarity": 0.0,
                "is_same": False,
                "status": "error",
                "message": f"Ошибка верификации: {str(e)}"
            }
        # Добавляем классы, которые есть в returned но нет в taken
    for cls_id_r in det_returned_by_class:
        if cls_id_r not in det_taken_by_class:
            verification_results[cls_id_r] = {
                "class_name": CLASS_MAPPING[cls_id_r],
                "similarity": 0.0,
                "is_same": False,
                "status": "extra_class",
                "message": f"Класс {cls_id_r} отсутствует во взятых инструментах"
            }

    return verification_results


def verificate_solo_objects(data_taken, data_returned):
    url_taken = data_taken["original_url"]
    url_returned = data_returned["original_url"]
    try:
        img_taken_src = Image.open(url_taken.replace('/static/', 'static/')).convert('RGB')
        # img_taken_src = cv2.imread(url_taken.replace('/static/', 'static/'))
        img_returned_src = Image.open(url_returned.replace('/static/', 'static/')).convert('RGB')
        # img_returned_src = cv2.imread(url_returned.replace('/static/', 'static/'))
    except Exception as e:
        print(f"Ошибка загрузки изображений: {e}")
        return []

    img_taken_pad = add_padding_to_square(img_taken_src, target_size=320)
    img_returned_pad = add_padding_to_square(img_returned_src, target_size=320)

    # x1_r, y1_r, x2_r, y2_r = map(int, bbox_r)
    # crop_returned = img_returned.crop((x1_r, y1_r, x2_r, y2_r))

    # Верифицируем обрезанные области
    try:
        # Сохраняем временные файлы для верификации
        rand_hash = uuid.uuid4().hex
        temp_taken_path = f"static/verification_solo/{rand_hash}_taken_crop.jpg"
        temp_returned_path = f"static/verification_solo/{rand_hash}_returned_crop.jpg"

        # Создаем временную директорию если нет
        os.makedirs("static/verification_solo", exist_ok=True)

        img_taken_pad.save(temp_taken_path)
        img_returned_pad.save(temp_returned_path)

        # Выполняем верификацию
        is_same, similarity = verify_images(img_taken_pad, img_returned_pad, threshold=THRESHOLD)
        verification_results = {
            "similarity": similarity,
            "is_same": is_same,
            "status": "verified"}
    except Exception as e:
        print(f"Ошибка обработки изображений: {e}")
        verification_results = {
            "error": f"Ошибка обработки: {str(e)}",
            "status": "error"
        }
    return verification_results