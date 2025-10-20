import os
import datetime as dt
import shutil
from glob import glob
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CocoDetection
from tqdm import tqdm
from trainers.trainers import BaseTrainer, collate_fn, log_dataset_statistics_to_tensorboard
from torchmetrics.detection import MeanAveragePrecision
from trainers.FasterRCNNTrainer import FasterRCNNTrainer
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.backbone_utils import BackboneWithFPN

# EfficientNet backbone вимагає бібліотеки timm
# Встановіть її командою: pip install timm
try:
    import timm
except ImportError:
    print("Попередження: бібліотеку 'timm' не знайдено. Бекбон EfficientNet буде недоступний.")
    print("Будь ласка, встановіть її командою: pip install timm")
    timm = None

# Словник з конфігураціями backbone: назва, рекомендований розмір (ширина, висота) та опис
BACKBONE_CONFIGS = {
    '1': ('resnet50', (800, 800), "ResNet-50 (стандартний, збалансований)"),
    '2': ('tf_efficientdet_d0', (512, 512), "EfficientDet-D0 (найлегший)"),
    '3': ('tf_efficientdet_d1', (640, 640), "EfficientDet-D1 (кращий баланс швидкість/точність)"),
    '4': ('tf_efficientdet_d2', (768, 768), "EfficientDet-D2"),
    '5': ('tf_efficientdet_d3', (896, 896), "EfficientDet-D3"),
    '6': ('tf_efficientdet_d4', (1024, 1024), "EfficientDet-D4"),
    '7': ('tf_efficientdet_d5', (1280, 1280), "EfficientDet-D5 (вища точність, повільніший)"),
    '8': ('tf_efficientdet_d6', (1536, 1536), "EfficientDet-D6"),
    '9': ('tf_efficientdet_d7', (1536, 1536), "EfficientDet-D7 (найвища точність)"),
}

# --- Допоміжний клас для трансформацій ---
class DetectionTransforms:
    def __init__(self, is_train=True, cat_id_map=None, imgsz=None):
        self.cat_id_map = cat_id_map
        transforms_list = []
        if imgsz:
            # imgsz це (ширина, висота), а T.Resize очікує (висота, ширина)
            height, width = imgsz[1], imgsz[0]
            transforms_list.append(T.Resize((height, width)))
        transforms_list.append(T.ToTensor())
        self.transforms = T.Compose(transforms_list)

    def __call__(self, image, target):
        if not target: # Обробка зображень без анотацій
            transformed_target = {
                'boxes': torch.empty((0, 4), dtype=torch.float32),
                'labels': torch.empty(0, dtype=torch.int64)
            }
            image = self.transforms(image)
            return image, transformed_target

        boxes = [ann['bbox'] for ann in target]
        boxes = torch.tensor(boxes, dtype=torch.float32)
        # Перетворення з [x, y, width, height] в [x1, y1, x2, y2]
        if boxes.numel() > 0:
            boxes[:, 2:] += boxes[:, :2]

        labels = torch.tensor([self.cat_id_map[ann['category_id']] for ann in target], dtype=torch.int64)

        transformed_target = { 'boxes': boxes, 'labels': labels }
        image = self.transforms(image)
        return image, transformed_target

# --- Тренер для FCOS ---
class FCOSTrainer(BaseTrainer):
    """Керує процесом навчання моделі FCOS з вибором режиму навчання."""

    def __init__(self, training_params, dataset_dir):
        super().__init__(training_params, dataset_dir)
        self.training_mode = None
        self.backbone_type = None
        self.image_size = None

    def _get_model_name(self):
        """Повертає повну назву моделі для логування."""
        if not self.backbone_type:
            return "FCOS"
            
        backbone_str = "ResNet-50"
        if 'efficientdet' in self.backbone_type:
            backbone_str = self.backbone_type.replace('tf_efficientdet_d', 'EfficientDet-D')
        
        mode_map = {
            'head_only': 'Fine-tune (Head)',
            'head_fpn': 'Fine-tune (Head+FPN)',
            'full': 'Full Training'
        }
        mode_str = mode_map.get(self.training_mode, "Unknown Mode")
        return f"FCOS ({backbone_str}) - {mode_str}"

    def _select_configuration(self):
        """Запитує у користувача backbone та режим навчання для FCOS."""
        print("\n   Оберіть 'хребет' (backbone) для FCOS:")
        for key, (_, _, description) in BACKBONE_CONFIGS.items():
            print(f"     {key}: {description}")

        while self.backbone_type is None:
            choice = input(f"   Ваш вибір backbone (1-{len(BACKBONE_CONFIGS)}): ").strip()
            if choice in BACKBONE_CONFIGS:
                self.backbone_type, self.image_size, desc = BACKBONE_CONFIGS[choice]
                print(f"✅ Обрано backbone: {desc.split(' (')[0]} з розміром зображення {self.image_size}")
                if 'efficientdet' in self.backbone_type and timm is None:
                    print("❌ Помилка: бібліотека 'timm' не встановлена. Оберіть інший backbone.")
                    self.backbone_type = None # Скидаємо вибір
                    continue
            else:
                print(f"   ❌ Невірний вибір. Будь ласка, введіть число від 1 до {len(BACKBONE_CONFIGS)}.")

        print("\n   Оберіть режим навчання для FCOS:")
        print("     1: Fine-tuning (навчати тільки 'голову', найшвидше, рекомендовано для старту)")
        if self.backbone_type == 'resnet50':
             print("     2: Fine-tuning (навчати 'голову' та FPN, збалансований варіант)")
        print("     3: Full training (навчати всю модель, найдовше, потенційно найкраща точність)")

        while self.training_mode is None:
            prompt_options = "1, 2 або 3" if self.backbone_type == 'resnet50' else "1 або 3"
            choice = input(f"   Ваш вибір режиму ({prompt_options}): ").strip()
            if choice == '1':
                print("✅ Обрано режим: Fine-tuning (тільки 'голова').")
                self.training_mode = 'head_only'
            elif choice == '2' and self.backbone_type == 'resnet50':
                print("✅ Обрано режим: Fine-tuning ('голова' + FPN).")
                self.training_mode = 'head_fpn'
            elif choice == '3':
                print("✅ Обрано режим: Full training (вся модель).")
                self.training_mode = 'full'
            else:
                print(f"   ❌ Невірний вибір. Будь ласка, введіть один з варіантів: {prompt_options}.")

    def start_or_resume_training(self, dataset_stats):
        if self.training_mode is None or self.backbone_type is None:
            self._select_configuration()

        imgsz = self.image_size
        print(f"\n--- Запуск тренування для {self._get_model_name()} ---")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔌 Обрано пристрій для навчання: {str(device).upper()}")

        print(f"🖼️ Розмір зображень для навчання буде змінено на {imgsz[0]}x{imgsz[1]}.")

        project_dir = os.path.join(self.params.get('project', 'runs/fcos'), f"{self.backbone_type}_{self.training_mode}")
        epochs = self.params.get('epochs', 25)
        batch_size = self.params.get('batch', 8)
        learning_rate = self.params.get('lr', 0.0001)
        step_size = self.params.get('lr_scheduler_step_size', 8)
        gamma = self.params.get('lr_scheduler_gamma', 0.1)
        self.accumulation_steps = self.params.get('accumulation_steps', 1)

        train_loader, val_loader, num_classes = self._prepare_dataloaders(batch_size, imgsz)
        print(f"📊 Знайдено {num_classes} класів. Навчання моделі для їх розпізнавання.")

        model = self._get_model(num_classes).to(device)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4)
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
            model, optimizer, start_epoch, best_map, lr_scheduler = self._load_checkpoint(
                checkpoint_path, model, optimizer, device, lr_scheduler
            )
            print(f"🚀 Відновлення навчання з {start_epoch}-ї епохи.")

        print(f"\n🚀 Розпочинаємо тренування на {epochs} епох...")
        for epoch in range(start_epoch, epochs):
            global_step = self._train_one_epoch(model, optimizer, train_loader, device, epoch, writer, global_step)
            val_map = self._validate_one_epoch(model, val_loader, device)

            lr_scheduler.step()

            print(f"Epoch {epoch + 1}/{epochs} | Validation mAP: {val_map:.4f} | Current LR: {lr_scheduler.get_last_lr()[0]:.6f}")

            writer.add_scalar('Validation/mAP', val_map, epoch)
            writer.add_scalar('LearningRate/Main', lr_scheduler.get_last_lr()[0], epoch)

            is_best = val_map > best_map
            if is_best:
                best_map = val_map

            self.save_checkpoint({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_map': best_map,
                'lr_scheduler_state_dict': lr_scheduler.state_dict()
            }, is_best, run_dir)

        writer.close()
        print("\n🎉 Навчання успішно завершено!")

        best_model_path = os.path.join(run_dir, "best_model.pth")
        final_path = None
        if os.path.exists(best_model_path):
            final_path = f"Final-{self._get_model_name().replace(' - ', '_').replace('(', '').replace(')', '')}-best.pth"
            shutil.copy(best_model_path, final_path)
            print(f"\n✅ Найкращу модель скопійовано у файл: {final_path} (mAP: {best_map:.4f})")

        summary = {
            "model_name": self._get_model_name(),
            "image_count": dataset_stats.get("image_count", "N/A"),
            "negative_count": dataset_stats.get("negative_count", "N/A"),
            "class_count": dataset_stats.get("class_count", num_classes),
            "image_size": self.image_size,
            "best_map": f"{best_map:.4f}",
            "best_model_path": final_path,
            "hyperparameters": self.params
        }
        return summary

    def _prepare_dataloaders(self, batch_size, imgsz=None):
        """Готує завантажувачі даних на основі COCO-формату."""
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

    def _get_model(self, num_classes):
        """Завантажує модель FCOS, адаптує її голову та налаштовує градієнти."""
        print(f"🔧 Створення моделі: {self._get_model_name()}")

        if 'efficientdet' in self.backbone_type:
            backbone_timm = timm.create_model(
                self.backbone_type,
                features_only=True,
                out_indices=(2, 3, 4),
                pretrained=True
            )
            in_channels_list = backbone_timm.feature_info.channels()
            
            backbone = BackboneWithFPN(
                backbone_timm,
                return_layers={'2': '0', '3': '1', '4': '2'},
                in_channels_list=in_channels_list,
                out_channels=256,
                extra_blocks=models.detection.fpn.LastLevelP6P7(256, 256)
            )
            model = models.detection.FCOS(backbone, num_classes=num_classes)
        else: # resnet50
            model = models.detection.fcos_resnet50_fpn(weights=models.detection.FCOS_ResNet50_FPN_Weights.DEFAULT)

        if self.training_mode == 'head_only':
            print("❄️ Заморожування backbone та FPN. Навчання тільки 'голови'.")
            for param in model.backbone.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True

        elif self.training_mode == 'head_fpn' and self.backbone_type == 'resnet50':
            print("❄️ Заморожування backbone. Навчання FPN та 'голови'.")
            for name, param in model.backbone.named_parameters():
                if 'fpn' not in name:
                    param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True

        elif self.training_mode == 'full':
            print("🔥 Усі ваги моделі розморожено для повного навчання.")
            for param in model.parameters():
                param.requires_grad = True

        in_channels = model.head.classification_head.conv[0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head.cls_logits = torch.nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        model.head.classification_head.num_classes = num_classes

        return model

    def _train_one_epoch(self, model, optimizer, data_loader, device, epoch, writer, global_step):
        model.train()
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1} [Train]")
        optimizer.zero_grad()

        for i, (images, targets) in enumerate(progress_bar):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                print(f"⚠️ Виявлено нескінченний loss на епосі {epoch + 1}, кроці {i}. Пропускаємо крок. Loss: {losses.item()}")
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
            progress_bar.set_postfix(loss=losses.item())
        return global_step

    def _validate_one_epoch(self, model, data_loader, device):
        model.eval()
        metric = MeanAveragePrecision(box_format='xyxy').to(device)
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Validating")
            for images, targets in progress_bar:
                images = [img.to(device) for img in images]
                targets_for_metric = [{k: v.to(device) for k, v in t.items()} for t in targets]
                predictions = model(images)
                metric.update(predictions, targets_for_metric)
        try:
            mAP_dict = metric.compute()
            return mAP_dict['map'].item()
        except Exception as e:
            print(f"Помилка при обчисленні mAP: {e}")
            return 0.0

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

        if lr_scheduler:
            return model, optimizer, start_epoch, best_map, lr_scheduler
        else:
            return model, optimizer, start_epoch, best_map