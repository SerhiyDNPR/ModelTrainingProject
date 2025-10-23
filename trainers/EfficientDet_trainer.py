# trainers/EfficientDet_trainer.py

import os
import datetime as dt
import shutil
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
import torchvision.transforms.functional as F # Потрібно для кастомних трансформацій
import random # Потрібно для аугментацій
from tqdm import tqdm
from trainers.trainers import BaseTrainer, collate_fn, log_dataset_statistics_to_tensorboard
from torchmetrics.detection import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter

# Спроба імпорту inputimeout для запиту з таймаутом
try:
    from inputimeout import inputimeout, TimeoutOccurred
except ImportError:
    # Заглушка, якщо бібліотека не встановлена
    class TimeoutOccurred(Exception):
        pass
    def inputimeout(prompt, timeout):
        print(prompt.replace(f"[автоматично '1' через {timeout}с]", "(Для роботи таймауту встановіть 'inputimeout')"))
        return input()

# EfficientDet вимагає сторонньої бібліотеки.
# Встановіть її командою: pip install effdet
try:
    from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
    from effdet.efficientdet import HeadNet
except ImportError:
    print("Помилка: бібліотеку 'effdet' не знайдено.")
    print("Будь ласка, встановіть її командою: pip install effdet")
    exit(1)

# Словник з конфігураціями моделей: назва та рекомендований розмір зображення (висота, ширина)
# ВАЖЛИВО: effdet очікує розмір у форматі (height, width)
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


class DetectionTransforms:
    """
    Власний клас трансформацій, що не залежить від FCOS_trainer.
    Виконує:
    1. Зміну розміру зображення до фіксованого `imgsz`.
    2. Масштабування bounding boxes відповідно до нового розміру.
    3. Горизонтальне віддзеркалення (аугментація) для навчання.
    4. Перетворення PIL Image в Tensor.
    5. Перетворення ID категорій COCO в послідовні ID (0, 1, 2...).
    """
    def __init__(self, is_train=False, cat_id_map=None, imgsz=None):
        self.is_train = is_train
        self.cat_id_map = cat_id_map

        if isinstance(imgsz, int):
            self.imgsz = (imgsz, imgsz) # (H, W)
        elif isinstance(imgsz, (tuple, list)) and len(imgsz) == 2:
            self.imgsz = imgsz # Очікуємо (H, W)
        else:
            raise ValueError("imgsz повинен бути int або (height, width) tuple/list.")

        if self.cat_id_map is None:
            raise ValueError("cat_id_map повинен бути наданий.")

    def __call__(self, image, target):
        # 1. Отримуємо оригінальний розмір
        w_orig, h_orig = image.size # PIL.size повертає (width, height)

        # 2. Змінюємо розмір зображення
        # F.resize очікує (H, W)
        image = F.resize(image, (self.imgsz[0], self.imgsz[1]))

        # 3. Горизонтальне віддзеркалення (аугментація)
        hflip = self.is_train and random.random() > 0.5
        if hflip:
            image = F.hflip(image)

        # 4. Перетворюємо зображення в тензор ([0, 255] -> [0.0, 1.0])
        image = F.to_tensor(image)

        # 5. Обробляємо цілі (targets)
        boxes = []
        labels = []

        # Розраховуємо коефіцієнти масштабування
        # self.imgsz = (H, W)
        w_scale = self.imgsz[1] / w_orig
        h_scale = self.imgsz[0] / h_orig

        if target: # target - це список dict'ів анотацій
            for ann in target:
                # Отримуємо послідовний ID класу
                label = self.cat_id_map.get(ann['category_id'])
                if label is None:
                    continue # Ігноруємо класи, яких немає в нашій мапі

                # COCO bbox = [x_min, y_min, width, height]
                x_min, y_min, w, h = ann['bbox']
                
                # Конвертуємо в [x_min, y_min, x_max, y_max]
                x_max = x_min + w
                y_max = y_min + h
                
                # Масштабуємо координати
                x_min_scaled = x_min * w_scale
                y_min_scaled = y_min * h_scale
                x_max_scaled = x_max * w_scale
                y_max_scaled = y_max * h_scale
                
                # Застосовуємо віддзеркалення до боксів, якщо воно було
                if hflip:
                    img_width_scaled = self.imgsz[1] # Ширина 'W'
                    # x_min стає (ширина - x_max)
                    # x_max стає (ширина - x_min)
                    x_max_new = img_width_scaled - x_min_scaled
                    x_min_new = img_width_scaled - x_max_scaled
                    x_min_scaled = x_min_new
                    x_max_scaled = x_max_new

                # Обрізаємо бокси, щоб вони були в межах зображення
                x_min_scaled = max(0, x_min_scaled)
                y_min_scaled = max(0, y_min_scaled)
                x_max_scaled = min(self.imgsz[1], x_max_scaled)
                y_max_scaled = min(self.imgsz[0], y_max_scaled)

                # Додаємо, тільки якщо бокс валідний (має площу)
                if x_max_scaled > x_min_scaled and y_max_scaled > y_min_scaled:
                    boxes.append([x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled])
                    labels.append(label)

        # Конвертуємо у тензори
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Обробка випадків, коли анотацій немає
        if not boxes.numel():
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            
        final_target = {}
        final_target["boxes"] = boxes
        final_target["labels"] = labels
        
        return image, final_target


def _create_model(num_classes, model_name='tf_efficientdet_d0', image_size=(512, 512), pretrained=True):
    """Створює модель EfficientDet з заданою конфігурацією."""
    config = get_efficientdet_config(model_name)
    config.num_classes = num_classes
    config.image_size = image_size # effdet очікує (height, width)

    model = EfficientDet(config, pretrained_backbone=pretrained)
    model.class_net = HeadNet(config, num_outputs=num_classes)
    return model

class EfficientDetTrainer(BaseTrainer):
    """Керує процесом навчання моделі EfficientDet."""

    def __init__(self, training_params, dataset_dir):
        super().__init__(training_params, dataset_dir)
        self.backbone_choice = None
        self.training_mode = None
        self.image_size = None # Буде зберігати обраний розмір зображення (H, W)

    def _select_configuration(self):
        """Запитує у користувача backbone та режим навчання."""
        print("\nБудь ласка, оберіть 'хребет' (backbone) для EfficientDet:")
        for key, (name, size) in BACKBONE_CONFIGS.items():
            model_id = name.replace('tf_efficientdet_', '').upper()
            print(f"  {key}: {model_id:<4} (рекомендований розмір: {size[0]}x{size[1]} [HxW])")

        while self.backbone_choice is None:
            choice = input(f"Ваш вибір (1-{len(BACKBONE_CONFIGS)}): ").strip()
            if choice in BACKBONE_CONFIGS:
                self.backbone_choice, self.image_size = BACKBONE_CONFIGS[choice]
                print(f"✅ Ви обрали: {self.backbone_choice} з розміром зображення {self.image_size} (H x W)")
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

        # Використовуємо розмір зображення, обраний користувачем (H, W)
        imgsz = self.image_size
        print(f"\n--- Запуск тренування для {self._get_model_name()} ---")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔌 Обрано пристрій для навчання: {str(device).upper()}")

        print(f"🖼️ Розмір зображень для навчання буде змінено на {imgsz[0]}x{imgsz[1]} (H x W).")

        project_dir = os.path.join(self.params.get('project', 'runs/efficientdet'), f"{self.backbone_choice}{self.training_mode}")
        epochs = self.params.get('epochs', 25)
        batch_size = self.params.get('batch', 8)
        learning_rate = self.params.get('lr', 0.0001)
        step_size = self.params.get('lr_scheduler_step_size', 8)
        gamma = self.params.get('lr_scheduler_gamma', 0.1)
        self.accumulation_steps = self.params.get('accumulation_steps', 1)

        train_loader, val_loader, num_classes = self._prepare_dataloaders(batch_size, imgsz=imgsz)
        print(f"📊 Знайдено {num_classes} класів. Навчання моделі для їх розпізнавання.")
        
        # --- ДІАГНОСТИКА: Перевірка трансформацій датасету (можна видалити після успішної перевірки) ---
        # ІМПОРТУЙТЕ visualize_batch_item СЮДИ АБО В utils.py
        # print("\n--- ДІАГНОСТИКА: Перевірка трансформацій датасету ---")
        # try:
        #     # Отримуємо один батч
        #     for images, targets in train_loader:
        #         break
        #     # Відображаємо лише перше зображення з батчу
        #     image_to_check = images[0]
        #     target_to_check = targets[0]
        #     visualize_batch_item(image_to_check, target_to_check, 
        #                          class_labels=list(range(num_classes)))
        #     answer = input("Трансформації виглядають коректно? (y/n): ").strip().lower()
        #     if answer not in ['y', 'Y', 'н', 'Н']:
        #         print("❌ Трансформації некоректні. Навчання зупинено.")
        #         return 
        #     else:
        #         print("✅ Трансформації коректні. Продовжуємо навчання.")
        # except Exception as e:
        #     print(f"❌ ПОМИЛКА ПРИ ВІЗУАЛІЗАЦІЇ: {e}. Продовжуємо, але будьте обережні.")
        # print("-----------------------------------------------------")
        # --- КІНЕЦЬ БЛОКУ ДІАГНОСТИКИ ---

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
            # Встановлюємо global_step на основі відновленої епохи
            global_step = start_epoch * len(train_loader)
            print(f"🚀 Відновлення навчання з {start_epoch}-ї епохи.")

        # --- Налаштування Warm-up ---
        warmup_epochs = 1  # Значення за замовчуванням
        if start_epoch == 0:
            try:
                prompt = f"\nВведіть кількість епох для 'прогріву' (warm-up) [автоматично '1' через 5с]: "
                user_input = inputimeout(prompt=prompt, timeout=5).strip()
                if user_input and user_input.isdigit() and int(user_input) > 0:
                    warmup_epochs = int(user_input)
                    print(f"✅ Встановлено {warmup_epochs} епох для прогріву.")
                else:
                    if user_input: # Якщо було введення, але некоректне
                         print(f"⚠️  Некоректне введення. Використовується значення за замовчуванням: {warmup_epochs} епоха.")
                    else: # Якщо просто натиснуто Enter
                         print(f"✅ Використовується значення за замовчуванням: {warmup_epochs} епоха.")

            except TimeoutOccurred:
                print(f"\nЧас на введення вичерпано. Використовується значення за замовчуванням: {warmup_epochs} епоха.")
            except Exception as e:
                print(f"\nПомилка при зчитуванні вводу ({e}). Використовується значення за замовчуванням: {warmup_epochs} епоха.")

            warmup_steps = warmup_epochs * len(train_loader)
            if warmup_steps > 0:
                print(f"🔥 Увімкнено 'прогрів' (warm-up) на {warmup_steps} кроків ({warmup_epochs} епох(и)).")
            else:
                 warmup_steps = 0 # На випадок порожнього датасету
        else:
            warmup_steps = 0  # Прогрів вже відбувся
            
        target_lr = learning_rate
        warmup_start_lr = 1e-7 # Початковий LR для прогріву
        # -----------------------------

        print(f"\n🚀 Розпочинаємо тренування на {epochs} епох...")
        for epoch in range(start_epoch, epochs):
            
            global_step = self._train_one_epoch(
                model, optimizer, train_loader, device, epoch, writer, global_step,
                target_lr=target_lr, 
                warmup_steps=warmup_steps, 
                warmup_start_lr=warmup_start_lr
            )

            val_map = self._validate_one_epoch(model, val_loader, device, imgsz=imgsz)
            
            if global_step > warmup_steps:
                lr_scheduler.step()

            current_display_lr = lr_scheduler.get_last_lr()[0] if global_step > warmup_steps else optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{epochs} | Validation mAP: {val_map:.4f} | Current LR: {current_display_lr:.6f}")
            writer.add_scalar('Validation/mAP', val_map, epoch)
            writer.add_scalar('LearningRate/Epoch', current_display_lr, epoch)

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
        
        best_model_path = os.path.join(run_dir, "best_model.pth")
        final_path = None
        if os.path.exists(best_model_path):
            final_path = f"Final-{self._get_model_name()}-best.pth"
            shutil.copy(best_model_path, final_path)
            print(f"\n✅ Найкращу модель скопійовано у файл: {final_path} (mAP: {best_map:.4f})")
        
        summary = { 
            "model_name": self._get_model_name(), 
            "best_map": f"{best_map:.4f}", 
            "best_model_path": final_path, 
            "hyperparameters": self.params 
        }
        return summary

    def _get_model(self, num_classes):
        """Завантажує та налаштовує модель EfficientDet."""
        print(f"🔧 Створення моделі: {self._get_model_name()}")
        model = _create_model(
            num_classes,
            self.backbone_choice,
            image_size=self.image_size, # (H, W)
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
        
        # Використовуємо наш новий, вбудований клас DetectionTransforms
        train_dataset = CocoDetection(root=train_img_dir, annFile=train_ann_file,
                                      transforms=DetectionTransforms(is_train=True, cat_id_map=cat_id_to_label, imgsz=imgsz))
        val_dataset = CocoDetection(root=val_img_dir, annFile=val_ann_file,
                                    transforms=DetectionTransforms(is_train=False, cat_id_map=cat_id_to_label, imgsz=imgsz))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)
        return train_loader, val_loader, num_classes

    def _train_one_epoch(self, model, optimizer, data_loader, device, epoch, writer, global_step,
                     target_lr, warmup_steps, warmup_start_lr):
        model.train()
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1} [Train]")
        optimizer.zero_grad()

        for i, (images, targets) in enumerate(progress_bar):
            
            # --- ЛОГІКА КЕРУВАННЯ LEARNING RATE ---
            if global_step < warmup_steps:
                lr_scale = global_step / warmup_steps
                new_lr = warmup_start_lr + lr_scale * (target_lr - warmup_start_lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
            
            elif global_step == warmup_steps:
                print(f"\n🔥 Warm-up завершено. Встановлено цільовий LR: {target_lr}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = target_lr
            
            images_tensor = torch.stack(images).to(device)
            boxes = [t['boxes'].to(device) for t in targets]
            cls_ids = [t['labels'].to(device) for t in targets]

            target_for_bench = {
                'bbox': boxes,
                'cls': cls_ids,
                'img_scale': torch.ones(len(images), device=device),
                'img_size': torch.tensor([i.shape[1:] for i in images], device=device)
            }
            loss_dict = model(images_tensor, target_for_bench)
            
            # 💡 ДОДАНО: Витягуємо індивідуальні значення втрат
            cls_loss = loss_dict['class_loss'].item()
            box_loss = loss_dict['box_loss'].item()
            
            losses = loss_dict['loss']

            if not torch.isfinite(losses):
                print(f"⚠️ Виявлено нескінченний loss. Пропускаємо крок.")
                optimizer.zero_grad()
                continue

            if self.accumulation_steps > 1:
                losses = losses / self.accumulation_steps

            losses.backward()
            
            current_lr = optimizer.param_groups[0]['lr']

            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(data_loader):
                # 💡 ЗМІНЕНО: Обрізання градієнтів (max_norm=0.5 для стабілізації)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) 
                
                optimizer.step()
                optimizer.zero_grad()
                
                display_loss = losses.item() * self.accumulation_steps
                
                writer.add_scalar('Train/Loss_step', display_loss, global_step)
                # 💡 ДОДАНО: Логування Classification Loss
                writer.add_scalar('Train/Classification_Loss', cls_loss, global_step)
                # 💡 ДОДАНО: Логування Box Regression Loss
                writer.add_scalar('Train/Box_Regression_Loss', box_loss, global_step)
                
                writer.add_scalar('LearningRate/Step', current_lr, global_step)
                
                # 💡 ЗМІНЕНО: Додаємо індивідуальні втрати до post-fix
                progress_bar.set_postfix(loss=display_loss, cls=cls_loss, box=box_loss, lr=f"{current_lr:.1E}")
                global_step += 1
            else:
                display_loss = losses.item() * self.accumulation_steps
                # 💡 ЗМІНЕНО: Додаємо індивідуальні втрати до post-fix
                progress_bar.set_postfix(loss=display_loss, cls=cls_loss, box=box_loss, lr=f"{current_lr:.1E}")

        return global_step

    def _validate_one_epoch(self, model, data_loader, device, imgsz):
        model.eval()
        pred_model = DetBenchPredict(model.model).to(device)
        pred_model.eval()

        metric = MeanAveragePrecision(box_format='xyxy').to(device)
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Validating")
            for images, targets in progress_bar:
                images_tensor = torch.stack(images).to(device)
                detections = pred_model(images_tensor)

                preds = []
                for det in detections:
                    preds.append({
                        'boxes': det[:, :4],
                        'scores': det[:, 4],
                        'labels': det[:, 5].int()
                    })
                
                targets_for_metric = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                metric.update(preds, targets_for_metric)
        try:
            mAP_dict = metric.compute()
            return mAP_dict['map'].item()
        except Exception as e:
            print(f"Помилка при обчисленні mAP: {e}")
            return 0.0

    from trainers.FasterRCNNTrainer import FasterRCNNTrainer
    _check_for_resume = FasterRCNNTrainer._check_for_resume_rcnn

    def _load_checkpoint(self, path, model, optimizer, device, lr_scheduler=None):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_map = checkpoint.get('best_map', 0.0)

        if lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
            try:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                print("✅ Стан планувальника LR успішно завантажено.")
            except Exception as e:
                print(f"⚠️ Не вдалося завантажити стан LR scheduler: {e}. Використовуються налаштування за замовчуванням.")

        return model, optimizer, start_epoch, best_map, lr_scheduler