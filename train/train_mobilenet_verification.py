import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.nn.functional import normalize
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


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


class VerificationDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train', add_padding=True, target_size=320):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.add_padding = add_padding
        self.target_size = target_size

        # Собираем пары изображений
        self.pairs = []
        self.labels = []

        # Структура папок: data_dir/class/image.jpg
        classes = os.listdir(data_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        # Создаем пары для обучения
        self._create_pairs()

    def _create_pairs(self):
        """Создает пары изображений для обучения"""
        classes = os.listdir(self.data_dir)

        for class_name in classes:
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            images = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Positive pairs (из одного класса)
            for i in range(len(images)):
                for j in range(i + 1, min(i + 3, len(images))):  # Ограничиваем количество пар
                    self.pairs.append((
                        os.path.join(class_path, images[i]),
                        os.path.join(class_path, images[j]),
                        1  # positive label
                    ))

            # Negative pairs (из разных классов)
            if "_own" in class_name:
                foreign_name = class_name.replace('_own', '_foreign')
                other_classes = [foreign_name]
            else:
                other_classes = [c for c in classes if c != class_name]
            n_negative_imgs = min(10, len(images) // 6)
            if other_classes:
                for img_name in images[:n_negative_imgs]:  # Берем несколько изображений
                    if len(other_classes) > 1:
                        other_class = random.choice(other_classes)
                    else:
                        n_rand = random.randint(0, len(other_classes) - 1)
                        other_class = random.choice([other_classes[n_rand], other_classes[0]])
                    other_class_path = os.path.join(self.data_dir, other_class)
                    other_images = [f for f in os.listdir(other_class_path)
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                    if other_images:
                        other_img = random.choice(other_images)
                        self.pairs.append((
                            os.path.join(class_path, img_name),
                            os.path.join(other_class_path, other_img),
                            0  # negative label
                        ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        # Добавляем паддинг если требуется
        if self.add_padding:
            img1 = add_padding_to_square(img1, target_size=self.target_size)
            img2 = add_padding_to_square(img2, target_size=self.target_size)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


class TripletDataset(Dataset):
    def __init__(self, data_dir, transform=None, add_padding=True, target_size=320, use_online_mining=False):
        self.data_dir = data_dir
        self.transform = transform
        self.add_padding = add_padding
        self.target_size = target_size

        # Собираем анкоры, позитивы и негативы
        self.triplets = []
        self._create_triplets()

    def _create_triplets(self):
        """Создает триплеты для обучения"""
        classes = os.listdir(self.data_dir)

        for class_name in classes:
            if "_foreign" in class_name:
                continue
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            images = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Для каждого изображения создаем триплеты
            for anchor_img in images:
                # Positive - другое изображение из того же класса
                positive_imgs = [img for img in images if img != anchor_img]
                if not positive_imgs:
                    continue

                positive_img = random.choice(positive_imgs)

                # Negative - изображение из другого класса
                if "_own" in class_name and random.random() < 0.3:
                    foreign_name = class_name.replace('_own', '_foreign')
                    other_classes = [foreign_name]
                else:
                    other_classes = [c for c in classes if c != class_name]

                if other_classes:
                    if len(other_classes) > 1:
                        other_class = random.choice(other_classes)
                    else:
                        n_rand = random.randint(0, len(other_classes) - 1)
                        other_class = random.choice([other_classes[n_rand], other_classes[0]])

                    # other_class = random.choice(other_classes)
                    other_class_path = os.path.join(self.data_dir, other_class)
                    negative_images = [f for f in os.listdir(other_class_path)
                                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                    if negative_images:
                        negative_img = random.choice(negative_images)

                        self.triplets.append((
                            os.path.join(class_path, anchor_img),
                            os.path.join(class_path, positive_img),
                            os.path.join(other_class_path, negative_img)
                        ))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path = self.triplets[idx]

        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')

        # Добавляем паддинг если требуется
        if self.add_padding:
            anchor = add_padding_to_square(anchor, target_size=self.target_size)
            positive = add_padding_to_square(positive, target_size=self.target_size)
            negative = add_padding_to_square(negative, target_size=self.target_size)

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative


class OnlineMiningDataset(Dataset):
    def __init__(self, data_dir, transform=None, add_padding=True, target_size=320):
        self.data_dir = data_dir
        self.transform = transform
        self.add_padding = add_padding
        self.target_size = target_size

        self.images = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        classes = os.listdir(self.data_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for class_name in classes:
            if "_foreign" in class_name:
                continue

            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            images = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            for img_name in images:
                self.images.append(os.path.join(class_path, img_name))
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')

        if self.add_padding:
            img = add_padding_to_square(img, target_size=self.target_size)

        if self.transform:
            img = self.transform(img)

        return img, label


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
            # Для EfficientNet добавляем adaptive pooling
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif backbone.startswith('vgg'):

            self.feature_extractor = base_model.features
            # self.channel_reduce = nn.Sequential(
            #     nn.Conv2d(512, 256, 1),  # Уменьшите количество каналов
            #     nn.BatchNorm2d(256),
            #     nn.ReLU(inplace=True)
            # )
            self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
            self.feature_dim = 512 * 7 * 7

        # Проекция в embedding space с улучшенной архитектурой
        self.embedding = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
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


class ImprovedTripletLoss(nn.Module):
    def __init__(self, margin=1.0, distance_metric='euclidean'):
        super(ImprovedTripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def _pairwise_distance(self, x, y):
        """Вычисляет расстояние между двумя наборами эмбеддингов"""
        if self.distance_metric == 'euclidean':
            return nn.functional.pairwise_distance(x, y)
        elif self.distance_metric == 'cosine':
            return 1 - nn.functional.cosine_similarity(x, y)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    def forward(self, anchor, positive, negative):
        pos_dist = self._pairwise_distance(anchor, positive)
        neg_dist = self._pairwise_distance(anchor, negative)

        # Triplet loss с мягким margin
        losses = torch.relu(pos_dist - neg_dist + self.margin)

        # Дополнительно: semi-hard negative mining
        semi_hard_mask = (neg_dist > pos_dist) & (neg_dist < pos_dist + self.margin)
        if semi_hard_mask.sum() > 0:
            return losses[semi_hard_mask].mean()

        return losses.mean()


class OnlineTripletLoss(nn.Module):
    def __init__(self, margin=1.0, distance_metric='euclidean'):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def _pairwise_distance(self, x, y):
        if self.distance_metric == 'euclidean':
            return torch.cdist(x.unsqueeze(0), y.unsqueeze(0)).squeeze(0)
        elif self.distance_metric == 'cosine':
            return 1 - nn.functional.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=2)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    def forward(self, embeddings, labels):
        # Вычисляем матрицу расстояний
        distance_matrix = self._pairwise_distance(embeddings, embeddings)

        # Для каждого анкора находим самый сложный позитив и негатив
        loss = 0.0
        num_valid_triplets = 0
        labels = labels.to(embeddings.device)

        for i in range(len(embeddings)):
            anchor_label = labels[i]

            # Positive mining
            positive_mask = (labels == anchor_label) & (torch.arange(len(labels), device=embeddings.device) != i)
            if not positive_mask.any():
                continue

            positive_distances = distance_matrix[i, positive_mask]
            hardest_positive = positive_distances.max()

            # Negative mining
            negative_mask = labels != anchor_label
            if not negative_mask.any():
                continue

            negative_distances = distance_matrix[i, negative_mask]
            hardest_negative = negative_distances.min()

            # Triplet loss
            triplet_loss = torch.relu(hardest_positive - hardest_negative + self.margin)
            if triplet_loss > 0:
                loss += triplet_loss
                num_valid_triplets += 1
        if num_valid_triplets == 0:
            print("num_valid_triplets = 0")
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return loss / max(num_valid_triplets, 1)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Евклидово расстояние
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)

        # Contrastive loss
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, output1, output2, label):
        # Косинусная схожесть (от -1 до 1)
        cosine_sim = nn.functional.cosine_similarity(output1, output2)

        # Преобразуем в вероятность (от 0 до 1)
        probability = (cosine_sim + 1) / 2

        # Binary cross entropy loss
        loss = nn.functional.binary_cross_entropy(probability, label)

        return loss


class VerificationTrainer:
    def __init__(self, model, device, checkpoint_dir='checkpoints', log_dir='runs/verification'):
        self.model = model
        self.device = device
        # self.writer = SummaryWriter(log_dir)
        self.checkpoint_dir = checkpoint_dir

        # Создаем директорию для чекпоинтов
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Статистика обучения
        self.train_losses = []
        self.val_accuracies = []
        self.learning_rates = []

        # История чекпоинтов
        self.checkpoint_history = []

    def save_checkpoint(self, epoch, optimizer, scheduler, val_acc, loss, is_best=False, additional_info=None):
        """Сохраняет чекпоинт модели"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_accuracy': val_acc,
            'loss': loss,
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'timestamp': time.time(),
            'git_hash': self._get_git_hash()  # опционально, для отслеживания версии кода
        }

        if additional_info:
            checkpoint.update(additional_info)

        # Сохраняем регулярный чекпоинт
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch:04d}.pth')
        torch.save(checkpoint, checkpoint_path)

        # Сохраняем как последний чекпоинт
        last_checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint_last.pth')
        torch.save(checkpoint, last_checkpoint_path)

        # Если это лучшая модель, сохраняем отдельно
        if is_best:
            best_checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_checkpoint_path)
            print(f'Новая лучшая модель сохранена с точностью: {val_acc:.4f}')

        # Сохраняем легковесную информацию о чекпоинте
        checkpoint_info = {
            'epoch': epoch,
            'val_accuracy': val_acc,
            'loss': loss,
            'path': checkpoint_path,
            'timestamp': checkpoint['timestamp'],
            'is_best': is_best
        }
        self.checkpoint_history.append(checkpoint_info)

        # Сохраняем историю чекпоинтов в JSON
        history_path = os.path.join(self.checkpoint_dir, 'checkpoint_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.checkpoint_history, f, indent=2)

        print(f'Чекпоинт эпохи {epoch} сохранен: {checkpoint_path}')

        # Очистка старых чекпоинтов (сохраняем только последние N)
        # self._cleanup_old_checkpoints(keep_last=5)

    def load_checkpoint(self, checkpoint_path, optimizer=None, scheduler=None):
        """Загружает чекпоинт модели"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Загружаем состояния
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Восстанавливаем историю обучения
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        self.learning_rates = checkpoint.get('learning_rates', [])

        print(f"Чекпоинт загружен: {checkpoint_path}")
        print(f"Эпоха: {checkpoint['epoch']}, Точность: {checkpoint['val_accuracy']:.4f}")

        return checkpoint['epoch'], checkpoint['val_accuracy']

    def _cleanup_old_checkpoints(self, keep_last=5):
        """Удаляет старые чекпоинты, оставляя только последние keep_last"""
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir)
                            if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]

        if len(checkpoint_files) <= keep_last:
            return

        # Сортируем по номеру эпохи
        checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))

        # Удаляем старые
        for old_checkpoint in checkpoint_files[:-keep_last]:
            old_path = os.path.join(self.checkpoint_dir, old_checkpoint)
            os.remove(old_path)
            print(f"Удален старый чекпоинт: {old_checkpoint}")

    def _get_git_hash(self):
        """Получает хэш git коммита (опционально)"""
        try:
            import subprocess
            return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        except:
            return "unknown"

    def train_triplet(self, train_loader, val_loader, num_epochs=50, lr=0.0001,
                      checkpoint_frequency=5, resume_from=None, use_online_mining=False):
        """Обучение с Triplet Loss с поддержкой чекпоинтов"""
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        if use_online_mining:
            criterion = OnlineTripletLoss(margin=0.5, distance_metric='euclidean')
        else:
            criterion = ImprovedTripletLoss(margin=0.5, distance_metric='cosine')

        start_epoch = 0
        best_val_acc = 0

        # Загрузка чекпоинта для возобновления обучения
        if resume_from:
            if resume_from == 'last':
                resume_path = os.path.join(self.checkpoint_dir, 'checkpoint_last.pth')
            elif resume_from == 'best':
                resume_path = os.path.join(self.checkpoint_dir, 'checkpoint_best.pth')
            else:
                resume_path = resume_from

            if os.path.exists(resume_path):
                start_epoch, best_val_acc = self.load_checkpoint(resume_path, optimizer, scheduler)
                start_epoch += 1
                print(f"Возобновление обучения с эпохи {start_epoch}")

        for epoch in range(start_epoch, num_epochs):
            # Обучение
            self.model.train()
            running_loss = 0.0

            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
            for batch_idx, batch_data in enumerate(pbar):
                if use_online_mining:
                    # Для online mining нужны все эмбеддинги батча
                    images, labels = batch_data
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    embeddings = self.model(images)
                    loss = criterion(embeddings, labels)
                else:
                    # Стандартные триплеты
                    anchor, positive, negative = batch_data
                    anchor = anchor.to(self.device)
                    positive = positive.to(self.device)
                    negative = negative.to(self.device)

                    optimizer.zero_grad()
                    anchor_emb, positive_emb, negative_emb = self.model(anchor, positive, negative)
                    loss = criterion(anchor_emb, positive_emb, negative_emb)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

            avg_loss = running_loss / len(train_loader)
            self.train_losses.append(avg_loss)

            # Валидация
            val_acc = self.validate(val_loader)
            self.val_accuracies.append(val_acc)

            current_lr = optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            # Обновление scheduler
            scheduler.step()

            print(f'Epoch {epoch + 1}: Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}')

            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc

            # Сохраняем чекпоинт
            if (epoch + 1) % checkpoint_frequency == 0 or is_best or (epoch + 1) == num_epochs:
                additional_info = {
                    'training_config': {
                        'loss_function': 'OnlineTripletLoss',
                        'backbone': self.model.backbone_name,
                        'input_size': self.model.input_size,
                        'embedding_dim': self.model.embedding[-2].out_features,
                        'online_mining': use_online_mining,
                        'padding_applied': True
                    }
                }

                self.save_checkpoint(
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    val_acc=val_acc,
                    loss=avg_loss,
                    is_best=is_best,
                    additional_info=additional_info
                )

    def get_checkpoint_info(self):
        """Возвращает информацию о доступных чекпоинтах"""
        checkpoints = []
        for f in os.listdir(self.checkpoint_dir):
            if f.endswith('.pth'):
                path = os.path.join(self.checkpoint_dir, f)
                checkpoint = torch.load(path, map_location='cpu')
                checkpoints.append({
                    'file': f,
                    'epoch': checkpoint['epoch'],
                    'val_accuracy': checkpoint['val_accuracy'],
                    'loss': checkpoint['loss'],
                    'timestamp': checkpoint.get('timestamp', 0)
                })

        return sorted(checkpoints, key=lambda x: x['epoch'])

    def validate(self, val_loader, threshold=0.7):
        """Валидация модели"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)

                output1, output2 = self.model(img1, img2)

                # Косинусная схожесть
                similarity = nn.functional.cosine_similarity(output1, output2)
                predictions = (similarity > threshold).float()

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        return accuracy

    def plot_training_history(self):
        """Визуализация процесса обучения"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        ax1.plot(self.train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        # Accuracy
        ax2.plot(self.val_accuracies)
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)

        # Learning Rate
        ax3.plot(self.learning_rates)
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('LR')
        ax3.set_yscale('log')
        ax3.grid(True)

        # Combined plot
        ax4.plot(self.train_losses, label='Train Loss')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(self.val_accuracies, 'r-', label='Val Accuracy')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4_twin.set_ylabel('Accuracy')
        ax4.set_title('Training Progress')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir, 'training_history.png'))
        plt.show()

    def find_optimal_threshold(self, val_loader):
        """Находит оптимальный порог для верификации"""
        self.model.eval()
        similarities = []
        labels_list = []

        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)

                output1, output2 = self.model(img1, img2)
                similarity = nn.functional.cosine_similarity(output1, output2)

                similarities.extend(similarity.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

        similarities = np.array(similarities)
        labels_list = np.array(labels_list)

        best_threshold = 0
        best_accuracy = 0

        for threshold in np.arange(0.1, 0.9, 0.01):
            predictions = (similarities > threshold).astype(float)
            accuracy = (predictions == labels_list).mean()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        print(f'Оптимальный порог: {best_threshold:.3f}, Точность: {best_accuracy:.3f}')

        # Сохраняем информацию о пороге
        threshold_info = {
            'optimal_threshold': best_threshold,
            'accuracy': best_accuracy,
            'timestamp': time.time()
        }

        with open(os.path.join(self.checkpoint_dir, 'threshold_info.json'), 'w') as f:
            json.dump(threshold_info, f, indent=2)

        return best_threshold


def main():
    # Настройки
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Увеличенный размер изображения
    INPUT_SIZE = 320
    use_online_mining = False

    # Аугментации для большего размера изображения
    train_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomResizedCrop(size=INPUT_SIZE, scale=(0.5, 1.2), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.7, 1.4), shear=(-15, 15, -15, 15)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.2),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Датасеты с паддингом
    if use_online_mining:
        train_dataset = OnlineMiningDataset(
            "/home/ubuntu/Projects/train/dataset/v1_split/train",
            transform=train_transform,
            add_padding=True,
            target_size=INPUT_SIZE
        )
    else:
        train_dataset = TripletDataset(
            "/home/ubuntu/Projects/train/dataset/v1_split/train",
            transform=train_transform,
            add_padding=True,
            target_size=INPUT_SIZE
        )
    val_dataset = VerificationDataset(
        "/home/ubuntu/Projects/train/dataset/v1_split/val",
        transform=val_transform,
        add_padding=True,
        target_size=INPUT_SIZE
    )

    # Даталоадеры
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Улучшенная модель с VGG19 и увеличенным размером входа
    model = ImprovedSiameseNetwork(
        embedding_dim=256,
        input_size=INPUT_SIZE,
        backbone='mobilenet_v3_small'  # Можно изменить на 'resnet101', 'efficientnet_b3'
    ).to(device)

    # Вывод информации о модели
    print(f"Модель: {model.backbone_name}")
    print(f"Размер входа: {model.input_size}x{model.input_size}")
    print(f"Размер эмбеддинга: {model.embedding[-2].out_features}")
    print(f"Количество параметров: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Применяется паддинг: Да")

    # Тренер с поддержкой чекпоинтов
    trainer = VerificationTrainer(model, device, checkpoint_dir='mobilenet_v3s_padded_training_checkpoints_v1',
                                  log_dir='runs/improved_triplet')

    # Показываем доступные чекпоинты (если есть)
    checkpoint_info = trainer.get_checkpoint_info()
    if checkpoint_info:
        print("Доступные чекпоинты:")
        for cp in checkpoint_info:
            print(f"  {cp['file']}: эпоха {cp['epoch']}, точность {cp['val_accuracy']:.4f}")

    # Обучение с улучшенным triplet loss
    print("Начало обучения с улучшенной архитектурой...")
    trainer.train_triplet(
        train_loader,
        val_loader,
        num_epochs=85,
        lr=0.0001,
        checkpoint_frequency=1,
        resume_from='best',  # Возобновить с последнего чекпоинта
        use_online_mining=use_online_mining
    )

    # Поиск оптимального порога
    optimal_threshold = trainer.find_optimal_threshold(val_loader)

    # Визуализация результатов
    trainer.plot_training_history()

    print("Обучение завершено!")
    print(f"Рекомендуемый порог для верификации: {optimal_threshold:.3f}")
    print(f"Использованная архитектура: {model.backbone_name}")
    print(f"Размер входного изображения: {model.input_size}x{model.input_size}")
    print(f"Применялся паддинг: Да")


if __name__ == "__main__":
    main()

    # nohup python train_resnet.py > training_resnet.log 2>&1 &
    # tail -f training_resnet.log
