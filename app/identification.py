import os
from collections import defaultdict

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.nn.functional import normalize

from utils import CLASS_MAPPING


# from train_identification_vgg16 import SiameseNetwork

class ImprovedSiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=512, input_size=320, backbone='resnet50'):
        super(ImprovedSiameseNetwork, self).__init__()
        self.input_size = input_size
        self.backbone_name = backbone

        # Выбор backbone архитектуры
        if backbone == 'resnet50':
            base_model = models.resnet50(pretrained=True)
            self.feature_dim = 2048
        elif backbone == 'resnet101':
            base_model = models.resnet101(pretrained=True)
            self.feature_dim = 2048
        elif backbone == 'efficientnet_b3':
            base_model = models.efficientnet_b3(pretrained=True)
            self.feature_dim = 1536
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=True)
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Извлекаем слои до полносвязных
        if backbone.startswith('resnet'):
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        elif backbone.startswith('efficientnet'):
            self.feature_extractor = base_model.features
            # Для EfficientNet добавляем adaptive pooling
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Проекция в embedding space с улучшенной архитектурой
        self.embedding = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        # Инициализация весов
        self._initialize_weights()

    def _initialize_weights(self):
        """Инициализация весов embedding слоев"""
        for m in self.embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_once(self, x):
        if self.backbone_name.startswith('resnet'):
            x = self.feature_extractor(x)
            x = x.view(x.size(0), -1)
        elif self.backbone_name.startswith('efficientnet'):
            x = self.feature_extractor(x)
            x = self.adaptive_pool(x)
            x = x.view(x.size(0), -1)

        x = self.embedding(x)
        x = normalize(x, p=2, dim=1)  # L2 нормализация
        return x

    def forward(self, input1, input2=None, input3=None):
        if input2 is None and input3 is None:
            # Режим извлечения признаков
            return self.forward_once(input1)
        elif input3 is None:
            # Сиамский режим (два входа)
            output1 = self.forward_once(input1)
            output2 = self.forward_once(input2)
            return output1, output2
        else:
            # Триплетный режим (три входа)
            output1 = self.forward_once(input1)
            output2 = self.forward_once(input2)
            output3 = self.forward_once(input3)
            return output1, output2, output3


def load_verification_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    INPUT_SIZE = 320
    # Загрузка модели
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = ImprovedSiameseNetwork(embedding_dim=512, input_size=INPUT_SIZE, backbone='resnet50').to(device)
    checkpoint = torch.load('models/checkpoint_epoch_0006.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, transform


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
verification_model, transform = load_verification_model()
THRESHOLD = 0.7

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
    url_taken = data_taken["original_url"]
    det_returned = data_returned["detections"]
    url_returned = data_returned["original_url"]
    try:
        img_taken = Image.open(url_taken.replace('/static/', 'static/')).convert('RGB')
        img_returned = Image.open(url_returned.replace('/static/', 'static/')).convert('RGB')
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
        x1_t, y1_t, x2_t, y2_t = map(int, bbox_t)

        # обрезаем изображение
        crop_taken = img_taken.crop((x1_t, y1_t, x2_t, y2_t))

        if cls_id_t not in det_returned_by_class:
            verification_results[cls_id_t] = {
                "class_name": CLASS_MAPPING[cls_id_t],
                "similarity": 0.0,
                "is_same": False,
                "status": "class_missing",
                "message": f"Класс {cls_id_t} отсутствует в возвращенных инструментах"
            }
            continue

        bbox_r = det_returned_by_class[cls_id_t]
        x1_r, y1_r, x2_r, y2_r = map(int, bbox_r)
        crop_returned = img_returned.crop((x1_r, y1_r, x2_r, y2_r))

        # Верифицируем обрезанные области
        try:
            # Сохраняем временные файлы для верификации
            temp_taken_path = f"temp/taken_crop_{cls_id_t}.jpg"
            temp_returned_path = f"temp/returned_crop_{cls_id_t}.jpg"

            # Создаем временную директорию если нет
            os.makedirs("temp", exist_ok=True)

            crop_taken.save(temp_taken_path)
            crop_returned.save(temp_returned_path)

            # Выполняем верификацию
            is_same, similarity = verify_images(crop_taken, crop_returned, threshold=THRESHOLD)
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
