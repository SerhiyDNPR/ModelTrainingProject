# trainers/EfficientDet_trainer.py

import os
import datetime as dt
import shutil
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from tqdm import tqdm
from trainers.trainers import BaseTrainer, collate_fn, log_dataset_statistics_to_tensorboard
from torchmetrics.detection import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter

# Використовуємо той самий клас трансформацій, що і для FCOS/RetinaNet
from trainers.FCOS_trainer import DetectionTransforms

# EfficientDet вимагає сторонньої бібліотеки.
# Встановіть її командою: pip install effdet
try:
    from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
    from effdet.efficientdet import HeadNet
except ImportError:
    print("Помилка: бібліотеку 'effdet' не знайдено.")
    print("Будь ласка, встановіть її командою: pip install effdet")
    exit(1)

# Словник з конфігураціями моделей: назва та рекомендований розмір зображення (ширина, висота)
BACKBONE_CONFIGS = {
    '1': ('tf_efficientdet_d0', (512, 512)),
    '2': ('tf_efficientdet_d1', (640, 640)),
    '3': ('tf_efficientdet_d2', (768, 768)),
    '4': ('tf_efficientdet_d3', (896, 896)),
    '5': ('tf_efficientdet_d4', (1024, 1024)),
    '6': ('tf_efficientdet_d5', (1280, 1280)),
    '7': ('tf_efficientdet_d6', (1536, 1536)),
    '8': ('tf_efficientdet_d7', (1536, 1536)),
}


def _create_model(num_classes, model_name='tf_efficientdet_d0', image_size=(512, 512), pretrained=True):
    """Створює модель EfficientDet з заданою конфігурацією."""
    config = get_efficientdet_config(model_name)
    config.num_classes = num_classes
    config.image_size = image_size

    model = EfficientDet(config, pretrained_backbone=pretrained)
    model.class_net = HeadNet(config, num_outputs=num_classes)
    return model

class EfficientDetTrainer(BaseTrainer):
    """Керує процесом навчання моделі EfficientDet."""

    def __init__(self, training_params, dataset_dir):
        super().__init__(training_params, dataset_dir)
        self.backbone_choice = None
        self.training_mode = None
        self.image_size = None # Буде зберігати обраний розмір зображення

    def _select_configuration(self):
        """Запитує у користувача backbone та режим навчання."""
        print("\nБудь ласка, оберіть 'хребет' (backbone) для EfficientDet:")
        for key, (name, size) in BACKBONE_CONFIGS.items():
            model_id = name.replace('tf_efficientdet_', '').upper()
            print(f"  {key}: {model_id:<4} (рекомендований розмір: {size[0]}x{size[1]})")

        while self.backbone_choice is None:
            choice = input(f"Ваш вибір (1-{len(BACKBONE_CONFIGS)}): ").strip()
            if choice in BACKBONE_CONFIGS:
                self.backbone_choice, self.image_size = BACKBONE_CONFIGS[choice]
                print(f"✅ Ви обрали: {self.backbone_choice} з розміром зображення {self.image_size}")
            else:
                print(f"❌ Невірний вибір. Будь ласка, введіть число від 1 до {len(BACKBONE_CONFIGS)}.")

        print("\n   Оберіть режим навчання:")
        print("     1: Fine-tuning (навчати тільки 'голову', швидше, рекомендовано)")
        print("     2: Full training (навчати всю модель, довше)")
        while self.training_mode is None:
            sub_choice = input("   Ваш вибір режиму (1 або 2): ").strip()
            if sub_choice == '1':
                self.training_mode = '_finetune'
                print("✅ Обрано режим: Fine-tuning.")
            elif sub_choice == '2':
                self.training_mode = '_full'
                print("✅ Обрано режим: Full training.")
            else:
                print("   ❌ Невірний вибір. Будь ласка, введіть 1 або 2.")

    def _get_model_name(self):
        if not self.backbone_choice:
            return "EfficientDet"
        backbone_str = self.backbone_choice.replace('tf_efficientdet_', '').upper()
        mode_str = "Fine-tune" if self.training_mode == '_finetune' else "Full"
        return f"EfficientDet ({backbone_str} {mode_str})"

    def start_or_resume_training(self, dataset_stats):
        if not self.backbone_choice or not self.training_mode:
            self._select_configuration()

        # Використовуємо розмір зображення, обраний користувачем
        imgsz = self.image_size
        print(f"\n--- Запуск тренування для {self._get_model_name()} ---")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔌 Обрано пристрій для навчання: {str(device).upper()}")

        print(f"🖼️ Розмір зображень для навчання буде змінено на {imgsz[0]}x{imgsz[1]}.")

        project_dir = os.path.join(self.params.get('project', 'runs/efficientdet'), f"{self.backbone_choice}{self.training_mode}")
        epochs = self.params.get('epochs', 25)
        batch_size = self.params.get('batch', 8)
        learning_rate = self.params.get('lr', 0.0001)
        step_size = self.params.get('lr_scheduler_step_size', 8)
        gamma = self.params.get('lr_scheduler_gamma', 0.1)
        self.accumulation_steps = self.params.get('accumulation_steps', 1)

        train_loader, val_loader, num_classes = self._prepare_dataloaders(batch_size, imgsz=imgsz)
        print(f"📊 Знайдено {num_classes} класів. Навчання моделі для їх розпізнавання.")

        base_model = self._get_model(num_classes)
        model = DetBenchTrain(base_model, create_labeler=True).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        run_name, checkpoint_path = self._check_for_resume(project_dir)
        start_epoch, best_map, global_step = 0, 0.0, 0

        run_dir = os.path.join(project_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard_logs'))

        print(f"📂 Результати будуть збережені в: {run_dir}")
        print(f"📈 Логи для TensorBoard будуть збережені в: {writer.log_dir}")
        log_dataset_statistics_to_tensorboard(train_loader.dataset, writer)

        if checkpoint_path:
            model.model, optimizer, start_epoch, best_map, lr_scheduler = self._load_checkpoint(
                checkpoint_path, model.model, optimizer, device, lr_scheduler
            )
            print(f"🚀 Відновлення навчання з {start_epoch}-ї епохи.")

        print(f"\n🚀 Розпочинаємо тренування на {epochs} епох...")
        for epoch in range(start_epoch, epochs):
            global_step = self._train_one_epoch(model, optimizer, train_loader, device, epoch, writer, global_step)
            val_map = self._validate_one_epoch(model, val_loader, device, imgsz=imgsz)
            lr_scheduler.step()

            print(f"Epoch {epoch + 1}/{epochs} | Validation mAP: {val_map:.4f} | Current LR: {lr_scheduler.get_last_lr()[0]:.6f}")
            writer.add_scalar('Validation/mAP', val_map, epoch)
            writer.add_scalar('LearningRate/Main', lr_scheduler.get_last_lr()[0], epoch)

            is_best = val_map > best_map
            if is_best:
                best_map = val_map

            self.save_checkpoint({
                'epoch': epoch + 1, 'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_map': best_map,
                'lr_scheduler_state_dict': lr_scheduler.state_dict()
            }, is_best, run_dir)

        writer.close()
        print("\n🎉 Навчання успішно завершено!")
        # ... (код для збереження результатів та повернення статистики) ...
        return {}

    def _get_model(self, num_classes):
        """Завантажує та налаштовує модель EfficientDet."""
        print(f"🔧 Створення моделі: {self._get_model_name()}")
        model = _create_model(
            num_classes,
            self.backbone_choice,
            image_size=self.image_size,
            pretrained=True
        )

        if self.training_mode == '_finetune':
            print("❄️ Заморожування backbone та FPN. Навчання тільки 'голови'.")
            for param in model.backbone.parameters():
                param.requires_grad = False
            for param in model.fpn.parameters():
                param.requires_grad = False
            for param in model.class_net.parameters():
                param.requires_grad = True
            for param in model.box_net.parameters():
                param.requires_grad = True
        else:
            print("🔥 Усі ваги моделі розморожено для повного навчання.")
            for param in model.parameters():
                param.requires_grad = True

        return model

    def _prepare_dataloaders(self, batch_size, imgsz=None):
        train_img_dir = os.path.join(self.dataset_dir, 'train')
        train_ann_file = os.path.join(self.dataset_dir, 'annotations', 'instances_train.json')
        val_img_dir = os.path.join(self.dataset_dir, 'val')
        val_ann_file = os.path.join(self.dataset_dir, 'annotations', 'instances_val.json')
        temp_dataset = CocoDetection(root=train_img_dir, annFile=train_ann_file)
        coco_cat_ids = sorted(temp_dataset.coco.cats.keys())
        cat_id_to_label = {cat_id: i for i, cat_id in enumerate(coco_cat_ids)}
        num_classes = len(coco_cat_ids)
        train_dataset = CocoDetection(root=train_img_dir, annFile=train_ann_file,
                                      transforms=DetectionTransforms(is_train=True, cat_id_map=cat_id_to_label, imgsz=imgsz))
        val_dataset = CocoDetection(root=val_img_dir, annFile=val_ann_file,
                                    transforms=DetectionTransforms(is_train=False, cat_id_map=cat_id_to_label, imgsz=imgsz))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)
        return train_loader, val_loader, num_classes

    def _train_one_epoch(self, model, optimizer, data_loader, device, epoch, writer, global_step):
        model.train()
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1} [Train]")
        optimizer.zero_grad()

        for i, (images, targets) in enumerate(progress_bar):
            # Перетворюємо PIL зображення в тензори та об'єднуємо в батч
            images_tensor = torch.stack(images).to(device)            
            # Переміщуємо кожен тензор з boxes та labels на потрібний пристрій
            boxes = [t['boxes'].to(device) for t in targets]
            cls_ids = [t['labels'].to(device) for t in targets]

            # Створюємо словник, одразу розміщуючи всі тензори на правильному пристрої
            target_for_bench = {
                'bbox': boxes,
                'cls': cls_ids,
                'img_scale': torch.ones(len(images), device=device),
                'img_size': torch.tensor([i.shape[1:] for i in images], device=device)
            }
            loss_dict = model(images_tensor, target_for_bench)
            losses = loss_dict['loss']

            if not torch.isfinite(losses):
                print(f"⚠️ Виявлено нескінченний loss. Пропускаємо крок.")
                continue

            if self.accumulation_steps > 1:
                losses = losses / self.accumulation_steps

            losses.backward()

            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(data_loader):
                optimizer.step()
                optimizer.zero_grad()
                display_loss = losses.item() * self.accumulation_steps if self.accumulation_steps > 1 else losses.item()
                writer.add_scalar('Train/Loss_step', display_loss, global_step)
                global_step += 1
                progress_bar.set_postfix(loss=display_loss)
            else:
                 display_loss = losses.item() * self.accumulation_steps if self.accumulation_steps > 1 else losses.item()
                 progress_bar.set_postfix(loss=display_loss)

        return global_step

    def _validate_one_epoch(self, model, data_loader, device, imgsz):
        model.eval()
        metric = MeanAveragePrecision(box_format='xyxy').to(device)
        
        # Створюємо трансформацію для валідації
        transform = T.Compose([T.Resize(imgsz), T.ToTensor()])

        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Validating")
            for images, targets in progress_bar:
                # Об'єднуємо тензори з DataLoader
                images_tensor = torch.stack(images).to(device)

                # Формуємо цілі (targets) у форматі, який очікує модель,
                # аналогічно до циклу навчання.
                boxes = [t['boxes'].to(device) for t in targets]
                cls_ids = [t['labels'].to(device) for t in targets]
                target_for_bench = {
                    'bbox': boxes,
                    'cls': cls_ids,
                    'img_scale': torch.ones(len(images), device=device),
                    'img_size': torch.tensor([i.shape[1:] for i in images], device=device)
                }

                # Тепер передаємо в модель і зображення, і цілі
                output = model(images_tensor, target_for_bench)
                detections = output['detections']

                # Конвертуємо передбачення та ground-truth у формат для torchmetrics
                preds = []
                for det in detections:
                    preds.append({
                        'boxes': det[:, :4],
                        'scores': det[:, 4],
                        'labels': det[:, 5].int()
                    })
                
                # Ground truth для метрики
                targets_for_metric = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                metric.update(preds, targets_for_metric)
        try:
            mAP_dict = metric.compute()
            return mAP_dict['map'].item()
        except Exception as e:
            print(f"Помилка при обчисленні mAP: {e}")
            return 0.0

    # Використовуємо методи з FasterRCNNTrainer для уніфікації
    from trainers.FasterRCNNTrainer import FasterRCNNTrainer
    _check_for_resume = FasterRCNNTrainer._check_for_resume_rcnn

    def _load_checkpoint(self, path, model, optimizer, device, lr_scheduler=None):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_map = checkpoint.get('best_map', 0.0)

        if lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            print("✅ Стан планувальника LR успішно завантажено.")

        return model, optimizer, start_epoch, best_map, lr_scheduler