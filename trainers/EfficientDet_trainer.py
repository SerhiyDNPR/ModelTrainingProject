# trainers/EfficientDet_trainer.py

import os
import datetime as dt
import shutil
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms.functional as F
import random
from tqdm import tqdm
from trainers.trainers import BaseTrainer, collate_fn, log_dataset_statistics_to_tensorboard
from torchmetrics.detection import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter

try:
    from inputimeout import inputimeout, TimeoutOccurred
except ImportError:
    class TimeoutOccurred(Exception):
        pass
    def inputimeout(prompt, timeout):
        # Якщо inputimeout не встановлено, використовуємо звичайний input
        # і додаємо повідомлення про необхідність встановлення.
        if 'автоматично' in prompt:
             print("⚠️ Для роботи таймауту встановіть 'pip install inputimeout'")
        return input(prompt)

try:
    from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
    from effdet.efficientdet import HeadNet
except ImportError:
    print("Помилка: бібліотеку 'effdet' не знайдено. Встановіть її: pip install effdet")
    exit(1)

# Видалено імпорти timm, які викликають проблеми

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
    def __init__(self, is_train=False, cat_id_map=None, imgsz=None):
        self.is_train = is_train
        self.cat_id_map = cat_id_map
        if isinstance(imgsz, int):
            self.imgsz = (imgsz, imgsz)
        elif isinstance(imgsz, (tuple, list)) and len(imgsz) == 2:
            self.imgsz = imgsz
        else:
            raise ValueError("imgsz повинен бути int або (height, width).")
        if self.cat_id_map is None:
            raise ValueError("cat_id_map повинен бути наданий.")

    def __call__(self, image, target):
        w_orig, h_orig = image.size
        image = F.resize(image, (self.imgsz[0], self.imgsz[1]))

        hflip = self.is_train and random.random() > 0.5
        if hflip:
            image = F.hflip(image)

        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        boxes, labels = [], []
        w_scale = self.imgsz[1] / w_orig
        h_scale = self.imgsz[0] / h_orig

        if target:
            for ann in target:
                label = self.cat_id_map.get(ann['category_id'])
                if label is None:
                    continue # Ігноруємо відсутні класи
                x_min, y_min, w, h = ann['bbox']
                if w < 0.1 or h < 0.1:
                    continue # Ігноруємо дуже малі бокси
                x_max, y_max = x_min + w, y_min + h
                x_min, x_max = x_min * w_scale, x_max * w_scale
                y_min, y_max = y_min * h_scale, y_max * h_scale

                if hflip:
                    img_w = self.imgsz[1]
                    x_min, x_max = img_w - x_max, img_w - x_min

                x_min = max(0, x_min); y_min = max(0, y_min)
                x_max = min(self.imgsz[1], x_max); y_max = min(self.imgsz[0], y_max)

                if x_max <= x_min or y_max <= y_min:
                    continue # Ігноруємо невалідні scaled бокси

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(label + 1)  # 1-based for effdet (клас 0 зарезервований для фону)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        final_target = {"boxes": boxes, "labels": labels}
        return image, final_target

def _create_model(num_classes, model_name='tf_efficientdet_d0', image_size=(512, 512), pretrained=True):
    config = get_efficientdet_config(model_name)
    config.num_classes = num_classes
    config.image_size = image_size
    # Налаштування Focal Loss (взято з config.py)
    config.label_smoothing = 0.01
    config.focal_loss_gamma = 1.5
    config.focal_loss_alpha = 0.75
    config.box_loss_weight = 50.0
    model = EfficientDet(config, pretrained_backbone=pretrained)
    model.class_net = HeadNet(config, num_outputs=num_classes)
    return model

class EfficientDetTrainer(BaseTrainer):
    def __init__(self, training_params, dataset_dir):
        super().__init__(training_params, dataset_dir)
        self.backbone_choice = None
        self.training_mode = None
        self.image_size = None

    def _select_configuration(self):
        print("\n   Оберіть 'хребет' (backbone) для EfficientDet:")
        for key, (name, size) in BACKBONE_CONFIGS.items():
            model_id = name.replace('tf_efficientdet_', '').upper()
            print(f"     {key}: {model_id:<4} (рекомендований розмір: {size[0]}x{size[1]} [H x W])")
        
        while self.backbone_choice is None:
            choice = input(f"   Ваш вибір backbone (1-{len(BACKBONE_CONFIGS)}): ").strip()
            if choice in BACKBONE_CONFIGS:
                self.backbone_choice, self.image_size = BACKBONE_CONFIGS[choice]
                print(f"✅ Обрано backbone: {self.backbone_choice} з розміром зображення {self.image_size} (H x W)")
            else:
                print(f"   ❌ Невірний вибір. Будь ласка, введіть число від 1 до {len(BACKBONE_CONFIGS)}.")

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

    def _get_model(self, num_classes):
        print(f"🔧 Створення моделі: {self._get_model_name()}")
        model = _create_model(
            num_classes,
            self.backbone_choice,
            image_size=self.image_size, 
            pretrained=True
        )
        if self.training_mode == '_finetune':
            for name, param in model.named_parameters():
                if name.startswith('backbone.'):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        return model

    def start_or_resume_training(self, dataset_stats):
        if self.backbone_choice is None or self.training_mode is None:
            self._select_configuration()

        imgsz = self.image_size
        print(f"\n--- Запуск тренування для {self._get_model_name()} ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        project_dir = os.path.join(self.params.get('project', 'runs/efficientdet'), f"{self.backbone_choice}{self.training_mode}")
        epochs = self.params.get('epochs', 30)
        batch_size = self.params.get('batch', 2)
        learning_rate = self.params.get('lr', 0.0005)
        self.accumulation_steps = self.params.get('accumulation_steps', 8)

        train_loader, val_loader, num_classes = self._prepare_dataloaders(batch_size, imgsz)
        
        # --- ВІДНОВЛЕНО: Виведення статистики датасету ---
        log_dataset_statistics_to_tensorboard(train_loader.dataset, SummaryWriter(log_dir=os.path.join(project_dir, 'temp_logs')))
        print(f"📊 Знайдено {num_classes} класів. Навчання моделі для їх розпізнавання.")
        # ---------------------------------------------------

        model = self._get_model(num_classes)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        run_name, checkpoint_path = self._check_for_resume(project_dir)
        start_epoch, best_map, global_step = 0, 0.0, 0
        
        run_dir = os.path.join(project_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard_logs'))
        
        model = DetBenchTrain(model).to(device)

        warmup_epochs = 1
        try:
            prompt = f"\nВведіть кількість епох для 'прогріву' (warm-up) [автоматично '{warmup_epochs}' через 10с]: "
            user_input = inputimeout(prompt=prompt, timeout=10).strip()
            if user_input and user_input.isdigit() and int(user_input) > 0:
                warmup_epochs = int(user_input)
                print(f"✅ Встановлено {warmup_epochs} епох для прогріву.")
            else:
                print(f"✅ Використовується значення за замовчуванням: {warmup_epochs} епоха.")
        except TimeoutOccurred:
            print(f"\nЧас на введення вичерпано. Використовується значення за замовчуванням: {warmup_epochs} епоха.")
        except Exception:
            print(f"\nВикористовується значення за замовчуванням: {warmup_epochs} епоха.")

        warmup_steps = warmup_epochs * len(train_loader)
        if warmup_steps > 0:
            print(f"🔥 Увімкнено 'прогрів' (warm-up) на {warmup_steps} кроків ({warmup_epochs} епох(и)).")
        
        print(f"\n🚀 Розпочинаємо тренування на {epochs} епох...")
        for epoch in range(start_epoch, epochs):
            # Передаємо model, optimizer, data_loader, device, epoch, writer, global_step
            global_step = self._train_one_epoch(model, optimizer, train_loader, device, epoch, writer, global_step, target_lr=learning_rate, warmup_steps=warmup_steps, warmup_start_lr=1e-7)
            
            # --- ВІДНОВЛЕНО: Вивід mAP у консоль ---
            val_map = self._validate_one_epoch(model, val_loader, device, imgsz) 
            print(f"\nEpoch {epoch+1}/{epochs} | Validation mAP: {val_map:.4f}")
            # -------------------------------------
            
            lr_scheduler.step()
            
            writer.add_scalar('Validation/mAP', val_map, epoch)

            is_best = val_map > best_map
            if is_best:
                best_map = val_map

            self.save_checkpoint({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_map': best_map,
                'lr_scheduler_state_dict': lr_scheduler.state_dict()
            }, is_best, run_dir)

        writer.close()
        
        summary = {
            "model_name": self._get_model_name(),
            "image_size": self.image_size,
            "best_map": f"{best_map:.4f}",
            "best_model_path": os.path.join(run_dir, "best_model.pth"),
            "hyperparameters": self.params
        }
        return summary

    def _prepare_dataloaders(self, batch_size, imgsz):
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

    # === ФІНАЛЬНИЙ МЕТОД _train_one_epoch ===
    def _train_one_epoch(self, model, optimizer, data_loader, device, epoch, writer, global_step, target_lr, warmup_steps, warmup_start_lr):
        model.train()
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1} [Train]")
        optimizer.zero_grad()
        
        for i, (images, targets) in enumerate(progress_bar):
            if global_step < warmup_steps:
                lr_scale = global_step / warmup_steps
                new_lr = warmup_start_lr + lr_scale * (target_lr - warmup_start_lr)
                for g in optimizer.param_groups: g['lr'] = new_lr

            images_tensor = torch.stack(images).to(device)
            
            # --- Створення цільових даних у форматі effdet (Dict з List[Tensor]) ---
            batch_bboxes = []
            batch_classes = []
            img_scales = []
            img_sizes = []
            
            for target_item in targets:
                if target_item['boxes'].numel() == 0:
                    bx = torch.zeros((0, 4), dtype=torch.float32, device=device)
                    cl = torch.zeros((0,), dtype=torch.int64, device=device)
                else:
                    bx = target_item['boxes'].to(device).float()
                    cl = target_item['labels'].to(device).long()
                
                batch_bboxes.append(bx)
                batch_classes.append(cl)
                
                img_scales.append(torch.tensor(1.0, dtype=torch.float32, device=device))
                img_sizes.append(torch.tensor(images_tensor[0].shape[1:], dtype=torch.float32, device=device))

            targets_for_bench = {
                'bbox': batch_bboxes, 
                'cls': batch_classes, 
                'img_scale': img_scales, 
                'img_size': img_sizes
            }

            if all(t.numel() == 0 for t in batch_bboxes):
                optimizer.zero_grad()
                continue
            
            try:
                loss_dict = model(images_tensor, targets_for_bench)
            except Exception as e:
                # Включає TypeError list indices must be integers or slices, not str
                print(f"[DEBUG] ❌ Помилка в моделі на batch {i}: {e}. Пропуск.")
                optimizer.zero_grad()
                continue
            
            cls_loss, box_loss = loss_dict['class_loss'].item(), loss_dict['box_loss'].item()
            loss = loss_dict['loss']

            if not torch.isfinite(loss):
                print("⚠️ Некоректний loss, пропускаємо.")
                optimizer.zero_grad()
                continue

            if self.accumulation_steps > 1:
                loss = loss / self.accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping
            
            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(data_loader):
                optimizer.step()
                optimizer.zero_grad()
                
                writer.add_scalar('Train/Loss_step', loss.item() * self.accumulation_steps, global_step)
                writer.add_scalar('Train/Classification_Loss', cls_loss, global_step)
                writer.add_scalar('Train/Box_Regression_Loss', box_loss, global_step)
                
                progress_bar.set_postfix(loss=loss.item() * self.accumulation_steps, cls=cls_loss, box=box_loss)
                global_step += 1
            else:
                 progress_bar.set_postfix(loss=loss.item() * self.accumulation_steps, cls=cls_loss, box=box_loss)

        return global_step
    # === КІНЕЦЬ _train_one_epoch ===


    # === ВАЛІДАЦІЙНИЙ МЕТОД ===
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
                    keep = det[:, 4] > 0.05 
                    preds.append({
                        'boxes': det[keep, :4],
                        'scores': det[keep, 4],
                        'labels': det[keep, 5].int()
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
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            
        return model, optimizer, start_epoch, best_map, lr_scheduler