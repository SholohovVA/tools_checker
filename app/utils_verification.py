from typing import List

import cv2
import numpy as np
import torchvision.models as models
from PIL import Image
from torch import nn
from torch.nn.functional import normalize


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
        elif backbone == 'vgg19':
            base_model = models.vgg19(pretrained=True)
            self.feature_dim = 512 * 7 * 7  # VGG19 features before classifier
        elif backbone == 'vgg16':
            base_model = models.vgg16(pretrained=True)
            self.feature_dim = 512 * 7 * 7  #
        elif backbone == 'mobilenet_v3_small':
            base_model = models.mobilenet_v3_small(pretrained=True)
            self.feature_extractor = base_model.features
            self.feature_dim = 576  # Для mobilenet_v3_small
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Добавьте это
        elif backbone == 'efficientnet_b0':  # Вместо b3
            base_model = models.efficientnet_b0(pretrained=True)
            self.feature_dim = 1280
        elif backbone == 'resnet18':  # Вместо resnet34/50
            base_model = models.resnet18(pretrained=True)
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Извлекаем слои до полносвязных
        if backbone.startswith('resnet'):
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        elif backbone.startswith('efficientnet'):
            self.feature_extractor = base_model.features
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif backbone.startswith('vgg'):
            self.feature_extractor = base_model.features
            self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
            self.feature_dim = 512 * 7 * 7

        # Проекция в embedding space с улучшенной архитектурой
        # self.embedding = nn.Sequential(
        #     nn.Dropout(0.3),
        #     nn.Linear(self.feature_dim, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.3),
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.3),
        #     nn.Linear(1024, embedding_dim),
        #     nn.BatchNorm1d(embedding_dim)
        # )
        self.embedding = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512)
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
        elif self.backbone_name.startswith('vgg'):
            x = self.feature_extractor(x)
            x = self.adaptive_pool(x)
            x = x.view(x.size(0), -1)
        elif self.backbone_name.startswith('mobilenet'):  # Добавлено
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


def create_masked_image(image: Image.Image, polygons: List) -> Image.Image:
    """
    Создает изображение с маской объекта по полигонам.
    """
    print(f"DEBUG: create_masked_image called with {len(polygons)} polygons")

    if not isinstance(image, np.ndarray):
        img_array = np.array(image)
    else:
        img_array = image.copy()

    # Создаем маску
    mask = np.zeros(img_array.shape[:2], dtype=np.uint8)

    # Фильтруем только валидные полигоны (минимум 3 точки)
    valid_polygons = []
    for i, polygon in enumerate(polygons):
        if len(polygon) >= 3:
            valid_polygons.append(polygon)
            print(f"DEBUG: Processing polygon {i} with {len(polygon)} points")
        else:
            print(f"DEBUG: Skipping polygon {i} with only {len(polygon)} points")

    if not valid_polygons:
        print("DEBUG: No valid polygons found, returning original image")
        return Image.fromarray(img_array)

    # Заполняем маску
    for polygon in valid_polygons:
        try:
            polygon_array = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask, [polygon_array], 255)
        except Exception as e:
            print(f"DEBUG: Error filling polygon: {e}")
            continue

    # Создаем результат с белым фоном
    white_bg = np.ones_like(img_array) * 255
    result = np.where(mask[:, :, None] == 255, img_array, white_bg)
    # Image.fromarray(
    return result.astype(np.uint8)



def preprocess_masks_on_image(img, seg_data, det_data):
    """
    Создает обрезанное изображение объекта на белом фоне используя маску сегментации.
    """
    print(f"DEBUG: preprocess_masks_on_image - image shape: {img.shape}")

    polygons = seg_data.get("polygons", [])
    print(f"DEBUG: Received {len(polygons)} polygons for processing")

    # Создаем маску
    img_mask = create_masked_image(img, polygons)

    # Обрезаем по bounding box детекции
    x0, y0, x1, y1 = map(int, det_data[:4])
    print(f"DEBUG: Crop coordinates: ({x0}, {y0}) to ({x1}, {y1})")

    # Проверяем границы
    h, w = img.shape[:2]
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))

    img_crop = img_mask[y0:y1, x0:x1]
    print(f"DEBUG: Final crop shape: {img_crop.shape}")

    return Image.fromarray(img_crop)

def add_padding_to_square(image: Image.Image, target_size: int = 320,
                          background_color: tuple = (255, 255, 255)) -> Image.Image:
    """
    Добавляет паддинг к изображению для преобразования в квадрат без искажений.

    Args:
        image: исходное PIL изображение
        target_size: размер целевого квадрата (default: 320)
        background_color: цвет фона для паддинга в формате RGB (default: белый)

    Returns:
        PIL Image квадратного формата target_size x target_size
    """
    # Получаем размеры исходного изображения
    width, height = image.size

    # Определяем коэффициент масштабирования
    scale = min(target_size / width, target_size / height)

    # Вычисляем новые размеры с сохранением пропорций
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Масштабируем изображение
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Создаем новое квадратное изображение с фоном
    squared_image = Image.new('RGB', (target_size, target_size), background_color)

    # Вычисляем позицию для вставки (центрирование)
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2

    # Вставляем масштабированное изображение в центр
    squared_image.paste(resized_image, (x_offset, y_offset))

    return squared_image
