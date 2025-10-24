# trainers/EfficientDet_trainer.py

import os
import datetime as dt
import shutil
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CocoDetection
from torchvision.models.detection.anchor_utils import AnchorGenerator
from tqdm import tqdm
from trainers.trainers import BaseTrainer, collate_fn, log_dataset_statistics_to_tensorboard
from torchmetrics.detection import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter

# Використовуємо той самий клас трансформацій, що і для FCOS
from trainers.FCOS_trainer import DetectionTransforms

try:
    import timm
except ImportError:
    raise ImportError("Бібліотека 'timm' не встановлена. Встановіть її за допомогою 'pip install timm'.")

from utils.backbone_factory import create_fpn_backbone

# Словник з конфігураціями backbone: назва, рекомендований розмір (ширина, висота), phi та опис
BACKBONE_CONFIGS = {
    '1': ('tf_efficientnet_b0', (512, 512), 0, "EfficientDet-D0 (найлегший)"),
    '2': ('tf_efficientnet_b1', (640, 640), 1, "EfficientDet-D1 (кращий баланс швидкість/точність)"),
    '3': ('tf_efficientnet_b2', (768, 768), 2, "EfficientDet-D2"),
    '4': ('tf_efficientnet_b3', (896, 896), 3, "EfficientDet-D3"),
    '5': ('tf_efficientnet_b4', (1024, 1024), 4, "EfficientDet-D4"),
    '6': ('tf_efficientnet_b5', (1280, 1280), 5, "EfficientDet-D5 (вища точність, повільніший)"),
    '7': ('tf_efficientnet_b6', (1536, 1536), 6, "EfficientDet-D6"),
    '8': ('tf_efficientnet_b7', (1536, 1536), 7, "EfficientDet-D7 (найвища точність)"),
}

class BiFPN(nn.Module):
    """Bidirectional Feature Pyramid Network (BiFPN) implementation."""
    def __init__(self, in_channels, out_channels, num_levels=5, num_layers=2):
        super(BiFPN, self).__init__()
        self.num_levels = num_levels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        # Learnable weights for feature fusion
        self.weights = nn.ParameterList([
            nn.Parameter(torch.ones(num_levels * 2 - 1) / (num_levels * 2 - 1)) for _ in range(num_layers)
        ])
        
        # Lateral convolutions to align channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels[i], out_channels, 1) for i in range(num_levels)
        ])
        
        # Separable convolutions for feature processing
        self.sep_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_levels)
        ])
        
        # Downsample and upsample for top-down and bottom-up pathways
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, inputs):
        """Forward pass for BiFPN."""
        assert len(inputs) == self.num_levels
        
        for layer in range(self.num_layers):
            # Lateral connections
            laterals = [self.lateral_convs[i](inputs[i]) for i in range(self.num_levels)]
            
            # Top-down pathway
            top_down = [laterals[-1]]  # Start with the highest level (P7)
            for i in range(self.num_levels - 2, -1, -1):
                weight = torch.softmax(self.weights[layer][:self.num_levels - 1], dim=0)
                upsampled = self.upsample(top_down[-1])
                # Ensure upsampled and lateral have the same spatial dimensions
                if upsampled.shape[2:] != laterals[i].shape[2:]:
                    upsampled = nn.functional.interpolate(upsampled, size=laterals[i].shape[2:], mode='nearest')
                fused = weight[i] * laterals[i] + (1 - weight[i]) * upsampled
                top_down.append(self.sep_convs[i](fused))
            top_down = top_down[::-1]  # Reverse to align with input order
            
            # Bottom-up pathway
            bottom_up = [top_down[0]]  # Start with the lowest level (P3)
            for i in range(1, self.num_levels):
                weight = torch.softmax(self.weights[layer][self.num_levels - 1:], dim=0)
                downsampled = self.downsample(bottom_up[-1])
                # Ensure downsampled and lateral have the same spatial dimensions
                if downsampled.shape[2:] != laterals[i].shape[2:]:
                    downsampled = nn.functional.interpolate(downsampled, size=laterals[i].shape[2:], mode='nearest')
                fused = weight[i - 1] * laterals[i] + (1 - weight[i - 1]) * downsampled
                bottom_up.append(self.sep_convs[i](fused))
            
            inputs = bottom_up  # Update inputs for the next BiFPN layer
        
        return inputs

class EfficientDetHead(nn.Module):
    """EfficientDet detection head for classification and box regression."""
    def __init__(self, in_channels, num_anchors, num_classes, num_layers=3, phi=0):
        super(EfficientDetHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        # Compound scaling: adjust channels based on phi (EfficientDet scaling factor)
        self.out_channels = int(64 * (1.1 ** phi))  # Base channels scaled by phi
        
        # Classification subnet
        cls_layers = []
        for _ in range(self.num_layers):
            cls_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, self.out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(self.out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = self.out_channels
        self.cls_subnet = nn.Sequential(*cls_layers)
        self.cls_predictor = nn.Conv2d(self.out_channels, num_anchors * num_classes, 3, padding=1)
        
        # Regression subnet
        reg_layers = []
        in_channels = self.out_channels  # Reset in_channels for regression subnet
        for _ in range(self.num_layers):
            reg_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, self.out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(self.out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = self.out_channels
        self.reg_subnet = nn.Sequential(*reg_layers)
        self.reg_predictor = nn.Conv2d(self.out_channels, num_anchors * 4, 3, padding=1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, features):
        cls_outputs = []
        box_outputs = []
        for feature in features:
            cls_logit = self.cls_subnet(feature)
            cls_logit = self.cls_predictor(cls_logit)
            box_reg = self.reg_subnet(feature)
            box_reg = self.reg_predictor(box_reg)
            
            # Reshape outputs: [N, num_anchors * num_classes, H, W] -> [N, H*W*num_anchors, num_classes]
            N, _, H, W = cls_logit.shape
            cls_logit = cls_logit.permute(0, 2, 3, 1).reshape(N, -1, self.num_classes)
            box_reg = box_reg.permute(0, 2, 3, 1).reshape(N, -1, 4)
            
            cls_outputs.append(cls_logit)
            box_outputs.append(box_reg)
        
        # Concatenate outputs across all feature levels
        cls_outputs = torch.cat(cls_outputs, dim=1)
        box_outputs = torch.cat(box_outputs, dim=1)
        
        return {'boxes': box_outputs, 'labels': cls_outputs}

class EfficientDet(nn.Module):
    """Custom EfficientDet model with BiFPN and detection head."""
    def __init__(self, backbone, num_classes, num_anchors, phi=0, num_bifpn_layers=2):
        super(EfficientDet, self).__init__()
        self.backbone = backbone
        self.bifpn = BiFPN(
            in_channels=[self.backbone.out_channels] * 5,  # Assuming 5 feature levels
            out_channels=256,
            num_layers=num_bifpn_layers
        )
        self.head = EfficientDetHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
            num_layers=3,
            phi=phi
        )
        
    def forward(self, images, targets=None):
        features = self.backbone(images)
        features = self.bifpn(list(features.values()))  # Convert OrderedDict to list
        predictions = self.head(features)
        
        if self.training:
            # Compute losses (requires custom loss function for training)
            return predictions  # Placeholder: actual loss computation needed
        else:
            return predictions

# --- Тренер для EfficientDet ---
class EfficientDetTrainer(BaseTrainer):
    """Керує процесом навчання моделі EfficientDet з вибором backbone та режиму."""
    
    def __init__(self, training_params, dataset_dir):
        super().__init__(training_params, dataset_dir)
        self.training_mode = None
        self.backbone_type = None
        self.image_size = None
        self.phi = 0  # Compound scaling factor

    def _select_configuration(self):
        """Запитує у користувача backbone та режим навчання для EfficientDet."""
        print("\n   Оберіть 'хребет' (backbone) для EfficientDet:")
        for key, (_, _, _, description) in BACKBONE_CONFIGS.items():
            print(f"     {key}: {description}")
        
        while self.backbone_type is None:
            choice = input(f"   Ваш вибір backbone (1-{len(BACKBONE_CONFIGS)}): ").strip()
            if choice in BACKBONE_CONFIGS:
                self.backbone_type, self.image_size, self.phi, desc = BACKBONE_CONFIGS[choice]
                print(f"✅ Обрано backbone: {desc.split(' (')[0]} з розміром зображення {self.image_size}, phi={self.phi}")
            else:
                print(f"   ❌ Невірний вибір. Будь ласка, введіть число від 1 до {len(BACKBONE_CONFIGS)}.")

        print("\n   Оберіть режим навчання:")
        print("     1: Fine-tuning (навчати тільки BiFPN і голову, швидше, рекомендовано)")
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
            return "EfficientDet"
            
        backbone_str = self.backbone_type.upper().replace('TF_', '').replace('_', '-')
        mode_name = "Fine-tune" if self.training_mode == '_finetune' else "Full"
        return f"EfficientDet ({backbone_str} {mode_name})"

    def _get_model(self, num_classes):
        """Завантажує модель EfficientDet з BiFPN і кастомною головою."""
        print(f"🔧 Створення моделі: {self._get_model_name()}")

        try:
            backbone = create_fpn_backbone(self.backbone_type, pretrained=True)
        except Exception as e:
            print(f"❌ Помилка при створенні backbone: {e}")
            raise

        anchor_generator = AnchorGenerator.from_config(
            config={
                "sizes": tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512]),
                "aspect_ratios": tuple([(0.5, 1.0, 2.0)] * 5),
            }
        )
        num_anchors = anchor_generator.num_anchors_per_location()[0]
        
        model = EfficientDet(
            backbone=backbone,
            num_classes=num_classes,
            num_anchors=num_anchors,
            phi=self.phi,
            num_bifpn_layers=2 + self.phi  # Scale BiFPN layers with phi
        )

        if self.training_mode == '_finetune':
            print("❄️ Заморожування ваг backbone. Навчання тільки BiFPN і голови.")
            for param in model.backbone.parameters():
                param.requires_grad = False
        else:
            print("🔥 Усі ваги моделі розморожено для повного навчання.")
            for param in model.parameters():
                param.requires_grad = True
        
        return model

    def start_or_resume_training(self, dataset_stats):
        if self.training_mode is None or self.backbone_type is None:
            self._select_configuration()

        imgsz = self.image_size
        print(f"\n--- Запуск тренування для {self._get_model_name()} ---")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔌 Обрано пристрій для навчання: {str(device).upper()}")
        
        print(f"🖼️ Розмір зображень для навчання буде змінено на {imgsz[0]}x{imgsz[1]}.")

        project_dir = os.path.join(self.params.get('project', 'runs/efficientdet'), f"{self.backbone_type}{self.training_mode}")
        epochs = self.params.get('epochs', 30)
        batch_size = self.params.get('batch', 2)
        learning_rate = self.params.get('lr', 0.0005)
        step_size = self.params.get('lr_scheduler_step_size', 10)
        gamma = self.params.get('lr_scheduler_gamma', 0.1)
        self.accumulation_steps = self.params.get('accumulation_steps', 8)

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
            
            predictions = model(images, targets)
            # Placeholder: Implement custom loss function for EfficientDet
            # For now, assume predictions include losses during training
            losses = sum(loss for loss in predictions.values()) if isinstance(predictions, dict) else predictions
            
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