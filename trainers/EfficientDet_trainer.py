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
import logging
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logging.basicConfig(level=logging.INFO)

try:
    from inputimeout import inputimeout, TimeoutOccurred
except ImportError:
    class TimeoutOccurred(Exception):
        pass
    def inputimeout(prompt, timeout):
        if 'автоматично' in prompt:
            print("⚠️ Для роботи таймауту встановіть 'pip install inputimeout'")
        return input(prompt)

try:
    from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
    from effdet.efficientdet import HeadNet
except ImportError:
    print("Помилка: бібліотеку 'effdet' не знайдено. Встановіть її: pip install effdet")
    exit(1)

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
    def __init__(self, is_train=False, cat_id_map=None, imgsz=None, dataset=None):
        self.is_train = is_train
        self.cat_id_map = cat_id_map
        self.dataset = dataset
        if isinstance(imgsz, int):
            self.imgsz = (imgsz, imgsz)
        elif isinstance(imgsz, (tuple, list)) and len(imgsz) == 2:
            self.imgsz = imgsz
        else:
            raise ValueError("imgsz повинен бути int або (height, width).")
        if self.cat_id_map is None:
            raise ValueError("cat_id_map повинен бути наданий.")

    def __call__(self, image, target, idx=None):
        w_orig, h_orig = image.size
        logging.debug(f"Оригінальні розміри зображення: {w_orig}x{h_orig}, цільовий розмір: {self.imgsz}")
        
        # Letterbox resize
        ratio = min(self.imgsz[0] / h_orig, self.imgsz[1] / w_orig)
        new_h, new_w = int(h_orig * ratio), int(w_orig * ratio)
        image = F.resize(image, (new_h, new_w))
        
        # Padding до цільового розміру
        pad_h = int((self.imgsz[0] - new_h) // 2)
        pad_w = int((self.imgsz[1] - new_w) // 2)
        pad_h_right = int(self.imgsz[0] - new_h - pad_h)
        pad_w_right = int(self.imgsz[1] - new_w - pad_w)
        pad = (pad_w, pad_h, pad_w_right, pad_h_right)
        logging.debug(f"Padding: {pad}")
        
        image = F.pad(image, pad, fill=0)

        hflip = self.is_train and random.random() > 0.5
        if hflip:
            image = F.hflip(image)

        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        boxes, labels = [], []
        w_scale = ratio
        h_scale = ratio
        removed_boxes = 0

        if target:
            for ann in target:
                label = self.cat_id_map.get(ann['category_id'])
                if label is None:
                    continue
                x_min, y_min, w, h = ann['bbox']
                if w < 0.1 or h < 0.1:
                    continue
                x_max, y_max = x_min + w, y_min + h
                x_min, x_max = x_min * w_scale + pad_w, x_max * w_scale + pad_w
                y_min, y_max = y_min * h_scale + pad_h, y_max * h_scale + pad_h

                if hflip:
                    img_w = self.imgsz[1]
                    x_min, x_max = img_w - x_max, img_w - x_min

                x_min = max(0, x_min); y_min = max(0, y_min)
                x_max = min(self.imgsz[1], x_max); y_max = min(self.imgsz[0], y_max)

                if x_max <= x_min or y_max <= y_min:
                    removed_boxes += 1
                    continue

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(label + 1)  # Potential issue to address in next step

        if removed_boxes > 0:
            logging.warning(f"Видалено {removed_boxes} невалідних боксів для зображення.")

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = target[0]['image_id'] if target else None

        final_target = {"boxes": boxes, "labels": labels, "image_id": image_id}
        return image, final_target

def _create_model(num_classes, model_name='tf_efficientdet_d0', image_size=(512, 512), pretrained=True):
    config = get_efficientdet_config(model_name)
    config.num_classes = num_classes
    config.image_size = image_size
    config.label_smoothing = 0.01
    config.focal_loss_gamma = 1.5
    config.focal_loss_alpha = 0.5
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
        self.accumulation_steps = training_params.get('accumulation_steps', 1)
        self.params['lr'] = training_params.get('lr', 0.0001)

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
        model = DetBenchTrain(model)
        if self.training_mode == '_finetune':
            for name, param in model.named_parameters():
                if name.startswith('model.backbone.'):
                    param.requires_grad = False
        return model

    def _prepare_dataloaders(self, batch_size, imgsz):
        train_img_dir = os.path.join(self.dataset_dir, 'train')
        train_ann_file = os.path.join(self.dataset_dir, 'annotations', 'instances_train.json')
        val_img_dir = os.path.join(self.dataset_dir, 'val')
        val_ann_file = os.path.join(self.dataset_dir, 'annotations', 'instances_val.json')

        temp_dataset = CocoDetection(root=train_img_dir, annFile=train_ann_file)
        coco_cat_ids = sorted(temp_dataset.coco.cats.keys())
        self.cat_id_to_label = {cat_id: i for i, cat_id in enumerate(coco_cat_ids)}
        self.label_to_cat_id = {i: cat_id for cat_id, i in self.cat_id_to_label.items()}
        num_classes = len(coco_cat_ids)
        logging.info(f"COCO Category IDs: {coco_cat_ids}")
        logging.info(f"cat_id_to_label: {self.cat_id_to_label}")
        logging.info(f"label_to_cat_id: {self.label_to_cat_id}")
        logging.info(f"Number of classes: {num_classes}")

        train_dataset = CocoDetection(root=train_img_dir, annFile=train_ann_file,
                                    transforms=DetectionTransforms(is_train=True, cat_id_map=self.cat_id_to_label, imgsz=imgsz))
        val_dataset = CocoDetection(root=val_img_dir, annFile=val_ann_file,
                                transforms=DetectionTransforms(is_train=False, cat_id_map=self.cat_id_to_label, imgsz=imgsz))

        def indexed_collate_fn(batch):
            batch_with_indices = [(img, tgt, i) for i, (img, tgt) in enumerate(batch)]
            images, targets = collate_fn([(img, tgt) for img, tgt, _ in batch_with_indices])
            indices = [idx for _, _, idx in batch_with_indices]
            return images, targets, indices

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=indexed_collate_fn, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=indexed_collate_fn, num_workers=0, pin_memory=True)
        return train_loader, val_loader, num_classes

    def _train_one_epoch(self, model, optimizer, data_loader, device, epoch, writer, global_step, target_lr, warmup_steps, warmup_start_lr):
        model.train()
        loss_total = 0
        loss_count = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}")

        for i, (images, targets, indices) in enumerate(progress_bar):  # Updated to unpack indices
            images = torch.stack(images).to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            
            # Log labels for debugging
            for idx, target in enumerate(targets):
                labels = target.get('labels', [])
                if len(labels) > 0:
                    logging.debug(f"Batch {i}, target {idx}: labels={labels.cpu().numpy().tolist()}")
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            if not math.isfinite(loss_value):
                logging.warning(f"Loss is {loss_value}, stopping training")
                return loss_total / loss_count if loss_count > 0 else 0, global_step

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            global_step += 1
            if global_step < warmup_steps:
                lr = warmup_start_lr + (target_lr - warmup_start_lr) * global_step / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            loss_total += loss_value
            loss_count += 1
            progress_bar.set_postfix(loss=loss_value, avg_loss=loss_total/loss_count, lr=optimizer.param_groups[0]['lr'])

            if writer:
                writer.add_scalar('Loss/train', loss_value, global_step)
                writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step)

        return loss_total / loss_count if loss_count > 0 else 0, global_step

    def _evaluate_coco(self, model, data_loader, device, val_ann_file):
        """Оцінює модель за допомогою COCO API."""
        model.eval()
        pred_model = DetBenchPredict(model.model).to(device)
        pred_model.eval()
        
        coco_gt = COCO(val_ann_file)
        coco_dt = []
        image_ids = []
        
        with torch.no_grad():
            for images, targets, indices in tqdm(data_loader, desc="COCO Evaluation"):
                images_tensor = torch.stack(images).to(device)
                detections = pred_model(images_tensor)
                
                for idx, (det, target, index) in enumerate(zip(detections, targets, indices)):
                    image_id = target.get('image_id')
                    if image_id is None:
                        logging.warning(f"Image ID is None for target at index {idx}, using dataset index {index}")
                        image_id = index
                    if isinstance(image_id, torch.Tensor):
                        image_id = image_id.item()
                    if image_id not in coco_gt.getImgIds():
                        logging.warning(f"Image ID {image_id} not found in COCO annotations")
                        continue
                    image_ids.append(image_id)
                    
                    keep = det[:, 4] > 0.05
                    boxes = det[keep, :4].cpu().numpy()
                    scores = det[keep, 4].cpu().numpy()
                    labels = det[keep, 5].int().cpu().numpy()
                    
                    logging.info(f"Model output labels for image {image_id}: {np.unique(labels)}")
                    
                    for box, score, label in zip(boxes, scores, labels):
                        category_id = self.label_to_cat_id.get(label, None)
                        if category_id is None:
                            logging.warning(f"Invalid category_id for label {label}")
                            continue
                        x_min, y_min, x_max, y_max = box
                        width = x_max - x_min
                        height = y_max - y_min
                        if width <= 0 or height <= 0:
                            logging.warning(f"Invalid bbox for image_id {image_id}: width={width}, height={height}")
                            continue
                        coco_dt.append({
                            'image_id': int(image_id),
                            'category_id': int(category_id),
                            'bbox': [float(x_min), float(y_min), float(width), float(height)],
                            'score': float(score)
                        })
        
        if not coco_dt:
            logging.warning("Не знайдено жодних передбачень для оцінки COCO.")
            return {'map': 0.0, 'map_50': 0.0, 'map_75': 0.0}
        
        logging.info(f"Generated {len(coco_dt)} predictions for {len(set(image_ids))} images")
        logging.info(f"Sample prediction: {coco_dt[:5] if coco_dt else 'No predictions'}")
        
        try:
            coco_dt = coco_gt.loadRes(coco_dt)
        except Exception as e:
            logging.error(f"Error loading predictions into COCO: {e}")
            return {'map': 0.0, 'map_50': 0.0, 'map_75': 0.0}
        
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = image_ids
        try:
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        except Exception as e:
            logging.error(f"Error during COCO evaluation: {e}")
            return {'map': 0.0, 'map_50': 0.0, 'map_75': 0.0}
        
        return {
            'map': coco_eval.stats[0],  # mAP@0.5:0.95
            'map_50': coco_eval.stats[1],  # mAP@0.5
            'map_75': coco_eval.stats[2]  # mAP@0.75
        }

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
                
                targets_for_metric = [{k: v.to(device) for k, v in t.items() if k != 'image_id'} for t in targets]
                metric.update(preds, targets_for_metric)
        
        try:
            mAP_dict = metric.compute()
            map_score = mAP_dict['map'].item()
        except Exception as e:
            print(f"Помилка при обчисленні mAP: {e}")
            map_score = 0.0

        val_ann_file = os.path.join(self.dataset_dir, 'annotations', 'instances_val.json')
        coco_metrics = self._evaluate_coco(model, data_loader, device, val_ann_file)
        
        writer = SummaryWriter(log_dir=os.path.join(self.params['project'], f"{self._get_model_name()}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"))
        writer.add_scalar('Validation/mAP', map_score, global_step=0)
        writer.add_scalar('Validation/COCO_mAP', coco_metrics['map'], global_step=0)
        writer.add_scalar('Validation/COCO_mAP_50', coco_metrics['map_50'], global_step=0)
        writer.add_scalar('Validation/COCO_mAP_75', coco_metrics['map_75'], global_step=0)
        writer.close()
        
        logging.info(f"COCO Metrics: mAP={coco_metrics['map']:.4f}, mAP@0.5={coco_metrics['map_50']:.4f}, mAP@0.75={coco_metrics['map_75']:.4f}")
        
        return map_score

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

    def _get_optimizer(self, model):
        if self.params.get('optimizer', 'AdamW') == 'SGD':
            return optim.SGD(
                model.parameters(),
                lr=self.params['lr'],
                momentum=self.params.get('momentum', 0.9),
                weight_decay=self.params.get('weight_decay', 0.0001)
            )
        else:
            return optim.AdamW(model.parameters(), lr=self.params['lr'])

    def start_or_resume_training(self, dataset_stats):
        import torch.optim.lr_scheduler as lr_scheduler
        import logging
        
        self._select_configuration()
        run_dir = os.path.join(self.params['project'], f"{self._get_model_name()}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=run_dir)
        
        batch_size = self.params['batch']
        imgsz = dataset_stats['image_size'] if dataset_stats.get('image_size') else self.image_size
        train_loader, val_loader, num_classes = self._prepare_dataloaders(batch_size, imgsz)
        
        log_dataset_statistics_to_tensorboard(train_loader.dataset, writer)
        
        model = self._get_model(num_classes).to(self.params['device'])
        optimizer = self._get_optimizer(model)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.params['epochs'], eta_min=1e-6)
        
        checkpoint_path, resume = self._check_for_resume(self.params['project'])
        start_epoch = 0
        best_map = 0.0
        global_step = 0
        
        if resume and checkpoint_path:
            model, optimizer, start_epoch, best_map, scheduler = self._load_checkpoint(
                checkpoint_path, model, optimizer, self.params['device'], scheduler
            )
            logging.info(f"Відновлено навчання з епохи {start_epoch}, найкращий mAP: {best_map}")
        
        warmup_steps = 1000
        warmup_start_lr = self.params['lr'] * 0.1
        target_lr = self.params['lr']
        
        for epoch in range(start_epoch, self.params['epochs']):
            logging.info(f"Починаємо епоху {epoch + 1}/{self.params['epochs']}")
            
            global_step = self._train_one_epoch(
                model, optimizer, train_loader, self.params['device'], epoch, writer,
                global_step, target_lr, warmup_steps, warmup_start_lr
            )
            
            map_score = self._validate_one_epoch(model, val_loader, self.params['device'], imgsz)
            logging.info(f"Епоха {epoch + 1}: mAP = {map_score}")
            writer.add_scalar('Validation/mAP', map_score, epoch)
            
            scheduler.step()
            
            is_best = map_score > best_map
            if is_best:
                best_map = map_score
                logging.info(f"Новий найкращий mAP: {best_map}")
            
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': scheduler.state_dict(),
                'best_map': best_map
            }
            self.save_checkpoint(state, is_best, run_dir)
            
            if 'patience' in self.params and epoch >= start_epoch + self.params['patience'] and map_score < best_map:
                logging.info(f"Раннє припинення: mAP не покращується протягом {self.params['patience']} епох")
                break
        
        writer.close()
        
        summary = {
            "model_name": self._get_model_name(),
            "image_count": dataset_stats.get("image_count", 0),
            "negative_count": dataset_stats.get("negative_count", 0),
            "class_count": dataset_stats.get("class_count", 0),
            "image_size": dataset_stats.get("image_size", imgsz),
            "best_map": best_map,
            "best_model_path": os.path.join(run_dir, "best_model.pth"),
            "hyperparameters": self.params
        }
        
        logging.info(f"Навчання завершено. Найкращий mAP: {best_map}")
        return summary