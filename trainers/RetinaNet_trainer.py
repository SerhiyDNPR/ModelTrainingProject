# trainers/RetinaNet_trainer.py

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
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import  RetinaNetHead
from tqdm import tqdm
from trainers.trainers import BaseTrainer, collate_fn, log_dataset_statistics_to_tensorboard
from torchmetrics.detection import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter

# Використовуємо той самий клас трансформацій, що і для FCOS
from trainers.FCOS_trainer import DetectionTransforms

from utils.backbone_factory import create_fpn_backbone

# Словник з конфігураціями backbone: назва, рекомендований розмір (ширина, висота) та опис
BACKBONE_CONFIGS = {
    '1': ('resnet50', (800, 800), "ResNet-50 (стандартний, збалансований)"),
    '2': ('swin_tiny_patch4_window7_224', (800, 800), "Swin-T (Tiny - Трансформер, швидкий)"),
    '3': ('swin_small_patch4_window7_224', (1024, 1024), "Swin-S (Small - Трансформер, збалансований)"),
}

# --- Тренер для RetinaNet ---
class RetinaNetTrainer(BaseTrainer):
    """Керує процесом навчання моделі RetinaNet з вибором backbone та режиму."""
    
    def __init__(self, training_params, dataset_dir):
        super().__init__(training_params, dataset_dir)
        self.training_mode = None
        self.backbone_type = None
        self.image_size = None

    def _select_configuration(self):
        print("\n   Оберіть 'хребет' (backbone) для RetinaNet:")
        for key, (_, _, description) in BACKBONE_CONFIGS.items():
            print(f"     {key}: {description}")
        
        while self.backbone_type is None:
            choice = input(f"   Ваш вибір backbone (1-{len(BACKBONE_CONFIGS)}): ").strip() # !!! Змінено межу
            if choice in BACKBONE_CONFIGS:
                self.backbone_type, self.image_size, desc = BACKBONE_CONFIGS[choice]
                print(f"✅ Обрано backbone: {desc.split(' (')[0]} з розміром зображення {self.image_size}")
                
                # --- УЗАГАЛЬНЕНА ПЕРЕВІРКА TIMM ---
                if 'swin' in self.backbone_type:
                    try:
                        import timm
                    except ImportError:
                        print("❌ Помилка: бібліотека 'timm' не встановлена. Оберіть інший backbone.")
                        self.backbone_type = None
                        continue
                # -----------------------------------
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
        if not self.backbone_type:
            return "RetinaNet"
            
        backbone_str = "ResNet-50"
        if 'swin' in self.backbone_type:
            backbone_str = self.backbone_type.upper().split('_')[0] + '-T' # Наприклад, SWIN-T
        
        mode_name = "Fine-tune" if self.training_mode == '_finetune' else "Full"
        return f"RetinaNet ({backbone_str} {mode_name})"

    def _get_model(self, num_classes):
        """Завантажує модель RetinaNet, адаптує її голову та заморожує ваги, якщо потрібно."""
        print(f"🔧 Створення моделі: {self._get_model_name()}")

        # --- ЛОГІКА ДЛЯ SWIN (BACKBONES З TIMM) ---
        if 'swin' in self.backbone_type:
            print(f"🔧 Створення моделі: {self._get_model_name()}")

            # create_fpn_backbone має обробляти swin
            backbone = create_fpn_backbone(self.backbone_type, pretrained=True, input_img_size = self.image_size[0])
            
            # --- СТВОРЕННЯ HEAD ТА МОДЕЛІ ---
            
            # ВИПРАВЛЕННЯ: Адаптація AnchorGenerator до кількості FPN рівнів (5 для Swin)
            anchor_level_sizes = [32, 64, 128, 256, 512]
            
            # Визначаємо параметри якірних боксів
            anchor_sizes = tuple(
                (x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in anchor_level_sizes
            )
            anchor_aspect_ratios = tuple([(0.5, 1.0, 2.0)] * len(anchor_level_sizes))

            # Створюємо AnchorGenerator
            anchor_generator = AnchorGenerator(
                sizes=anchor_sizes,
                aspect_ratios=anchor_aspect_ratios
            )
            
            in_channels = backbone.out_channels
            num_anchors = anchor_generator.num_anchors_per_location()[0]
            
            model = models.detection.RetinaNet(backbone, num_classes=num_classes, anchor_generator=anchor_generator)

        else: # 'resnet50' (використовує torchvision вбудований)
            model = models.detection.retinanet_resnet50_fpn_v2(weights=models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
            num_anchors = model.head.classification_head.num_anchors
            in_channels = model.backbone.out_channels
            new_head = RetinaNetHead(in_channels, num_anchors, num_classes)
            model.head = new_head

        # --- ЗАМОРОЖУВАННЯ ВАГ (Спільна логіка) ---
        if self.training_mode == '_finetune':
            print("❄️ Заморожування ваг backbone. Навчання тільки 'голови'.")
            # Заморожуємо параметри backbone, які є частиною моделі
            for param in model.backbone.parameters():
                param.requires_grad = False
        else:
            print("🔥 Усі ваги моделі розморожено для повного навчання.")
            for param in model.parameters():
                param.requires_grad = True
        
        return model

    # Решта коду файлу залишається без змін...
    def start_or_resume_training(self, dataset_stats):
        if self.training_mode is None or self.backbone_type is None:
            self._select_configuration()

        imgsz = self.image_size
        print(f"\n--- Запуск тренування для {self._get_model_name()} ---")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔌 Обрано пристрій для навчання: {str(device).upper()}")
        
        print(f"🖼️ Розмір зображень для навчання буде змінено на {imgsz[0]}x{imgsz[1]}.")

        project_dir = os.path.join(self.params.get('project', 'runs/retinanet'), f"{self.backbone_type}{self.training_mode}")
        epochs = self.params.get('epochs', 25)
        batch_size = self.params.get('batch', 8)
        learning_rate = self.params.get('lr', 0.0001)
        step_size = self.params.get('lr_scheduler_step_size', 8)
        gamma = self.params.get('lr_scheduler_gamma', 0.1)
        self.accumulation_steps = self.params.get('accumulation_steps', 1)

        if self.accumulation_steps > 1:
            effective_batch_size = batch_size * self.accumulation_steps
            print(f"🔄 Увімкнено накопичення градієнтів. Кроків: {self.accumulation_steps}. Ефективний batch_size: {effective_batch_size}")

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
            # Обробка несумісності чекпоінту
            try:
                model, optimizer, start_epoch, best_map, lr_scheduler = self._load_checkpoint(
                    checkpoint_path, model, optimizer, device, lr_scheduler
                )
                print(f"🚀 Відновлення навчання з {start_epoch}-ї епохи.")
            except RuntimeError as e:
                print(f"❌ Помилка завантаження чекпоінту: {e}. Навчання розпочнеться з нуля.")
                start_epoch, best_map, global_step = 0, 0.0, 0
        
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

            self._save_checkpoint({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_map': best_map,
                'lr_scheduler_state_dict': lr_scheduler.state_dict()
            }, is_best, run_dir)

        writer.close()
        print("\n🎉 Навчання успішно завершено!")
        
        best_model_path = os.path.join(run_dir, "best_model.pth")
        final_path = None
        if os.path.exists(best_model_path):
            final_path = f"Final-{self._get_model_name().replace(' (', '_').replace(')', '')}-best.pth"
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
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            if not torch.isfinite(losses):
                print(f"⚠️ Виявлено нескінченний loss на кроці {i}. Пропускаємо.")
                continue

            if self.accumulation_steps > 1:
                losses = losses / self.accumulation_steps
            losses.backward()
            
            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(data_loader):
                optimizer.step()
                optimizer.zero_grad()

                original_loss = losses.item() * self.accumulation_steps if self.accumulation_steps > 1 else losses.item()
                writer.add_scalar('Train/Loss_step', original_loss, global_step)
                global_step += 1
                progress_bar.set_postfix(loss=original_loss)
            else:
                original_loss = losses.item() * self.accumulation_steps if self.accumulation_steps > 1 else losses.item()
                progress_bar.set_postfix(loss=original_loss)

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