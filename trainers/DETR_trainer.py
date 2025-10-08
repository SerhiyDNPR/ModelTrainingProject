# DETR_trainer.py

import os
import datetime as dt
import shutil
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.datasets import CocoDetection
from trainers.trainers import BaseTrainer, log_dataset_statistics_to_tensorboard
from trainers.FasterRCNNTrainer import FasterRCNNTrainer
from transformers import DetrImageProcessor, AutoModelForObjectDetection
from torchmetrics.detection import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter

class CocoTransform:
    def __init__(self, cat_id_map):
        self.cat_id_map = cat_id_map

    def __call__(self, image, target):
        # Зберігаємо оригінальний розмір для постобробки
        orig_size = torch.tensor(image.size[::-1]) # h, w
        if not target:
            return image, {'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.int64), 'image_id': torch.tensor([-1]), 'orig_size': orig_size}

        image_id = torch.tensor([target[0]['image_id']])
        boxes = torch.tensor([ann['bbox'] for ann in target], dtype=torch.float32)
        boxes[:, 2:] += boxes[:, :2]
        labels = torch.tensor([self.cat_id_map[ann['category_id']] for ann in target], dtype=torch.int64)
        
        return image, {'boxes': boxes, 'labels': labels, 'image_id': image_id, 'orig_size': orig_size}

class DETRTrainer(BaseTrainer):
    """Керує процесом навчання моделей сімейства DETR."""

    def _get_project_name(self):
        return self.params.get('project', 'runs/detr')
    
    def _get_model_checkpoint(self):
        return "facebook/detr-resnet-50"

    def _initialize_processor(self):
        return DetrImageProcessor.from_pretrained(self._get_model_checkpoint())
        
    def _collate_fn_detr(self, batch):
        pixel_values = []
        labels = []
        for image, target in batch:
            if not target['boxes'].numel():
                continue
            individual_annotations = []
            for box, label_id in zip(target['boxes'], target['labels']):
                # Використовуємо self.id2label, який ініціалізується перед створенням DataLoader
                category_id = self.id2label[label_id.item()] 
                x_min, y_min, x_max, y_max = box.tolist()
                individual_annotations.append({
                    "image_id": target['image_id'].item(), "category_id": category_id,
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "area": (x_max - x_min) * (y_max - y_min)
                })
            
            encoding = self.image_processor(images=image, annotations=individual_annotations, return_tensors="pt")
            encoding["labels"][0]["orig_size"] = target["orig_size"]
            pixel_values.append(encoding["pixel_values"][0])
            labels.append(encoding["labels"][0])
        if not pixel_values:
             return None
        batch = {}
        batch["pixel_values"] = torch.stack(pixel_values)
        batch["labels"] = labels
        return batch

    def start_or_resume_training(self, dataset_stats):
        imgsz = dataset_stats.get('image_size')
        print(f"\n--- Запуск тренування для {self._get_model_name()} ---")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔌 Обрано пристрій для навчання: {str(device).upper()}")
        
        project_dir = self._get_project_name()
        epochs = self.params.get('epochs', 25)
        batch_size = self.params.get('batch', 2)
        learning_rate = self.params.get('lr', 1e-4)
        lr_backbone = self.params.get('lr_backbone', 1e-5)
        step_size = self.params.get('lr_scheduler_step_size', 15)
        gamma = self.params.get('lr_scheduler_gamma', 0.1)
        self.accumulation_steps = self.params.get('accumulation_steps', 1)
        if self.accumulation_steps > 1:
            effective_batch_size = batch_size * self.accumulation_steps
            print(f"🔄 Увімкнено накопичення градієнтів. Кроків: {self.accumulation_steps}. Ефективний batch_size: {effective_batch_size}")
        
        self.image_processor = self._initialize_processor()
        if imgsz:
            width, height = imgsz
            self.image_processor.size = {"height": height, "width": width}
            print(f"🖼️ Розмір зображень для навчання встановлено на {width}x{height}.")
        
        self.train_loader, self.val_loader, self.id2label = self._prepare_dataloaders(batch_size)
        train_loader, val_loader, id2label = self.train_loader, self.val_loader, self.id2label
        num_classes = len(id2label)
        print(f"📊 Знайдено {num_classes} класів.")
        
        model = self._get_model(id2label).to(device)
        
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": lr_backbone},
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=learning_rate, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        run_name, checkpoint_path = self._check_for_resume_rcnn(project_dir)
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
            if len(lr_scheduler.get_last_lr()) > 1:
                writer.add_scalar('LearningRate/Backbone', lr_scheduler.get_last_lr()[1], epoch)

            is_best = val_map > best_map
            if is_best: best_map = val_map

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
            final_path = f"Final-{self._get_model_name()}-best.pth"
            shutil.copy(best_model_path, final_path)
            print(f"\n✅ Найкращу модель скопійовано у файл: {final_path} (mAP: {best_map:.4f})")
        
        summary = {
            "model_name": self._get_model_name(),
            "image_count": dataset_stats.get("image_count", "N/A"),
            "negative_count": dataset_stats.get("negative_count", "N/A"),
            "class_count": dataset_stats.get("class_count", num_classes),
            "image_size": dataset_stats.get("image_size", "N/A"),
            "best_map": f"{best_map:.4f}",
            "best_model_path": final_path,
            "hyperparameters": self.params
        }
        return summary

    def _prepare_dataloaders(self, batch_size):
        train_img_dir = os.path.join(self.dataset_dir, 'train')
        train_ann_file = os.path.join(self.dataset_dir, 'annotations', 'instances_train.json')
        val_img_dir = os.path.join(self.dataset_dir, 'val')
        val_ann_file = os.path.join(self.dataset_dir, 'annotations', 'instances_val.json')
        temp_dataset = CocoDetection(root=train_img_dir, annFile=train_ann_file)
        sorted_cats = sorted(temp_dataset.coco.cats.values(), key=lambda x: x['id'])
        id2label = {i: cat['name'] for i, cat in enumerate(sorted_cats)}
        coco_cat_id_to_model_label = {cat['id']: i for i, cat in enumerate(sorted_cats)}
        transform = CocoTransform(coco_cat_id_to_model_label)
        train_dataset = CocoDetection(root=train_img_dir, annFile=train_ann_file, transforms=transform)
        val_dataset = CocoDetection(root=val_img_dir, annFile=val_ann_file, transforms=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn_detr, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self._collate_fn_detr, num_workers=8, pin_memory=True)
        return train_loader, val_loader, id2label
    
    def _get_model(self, id2label):
        model = AutoModelForObjectDetection.from_pretrained(
            self._get_model_checkpoint(), num_labels=len(id2label), id2label=id2label,
            label2id={name: id for id, name in id2label.items()},
            ignore_mismatched_sizes=True
        )
        return model

    def _train_one_epoch(self, model, optimizer, data_loader, device, epoch, writer, global_step):
        model.train()
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1} [Train]")
        optimizer.zero_grad()
        
        for i, batch in enumerate(progress_bar):
            if batch is None:
                continue
            
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch.pop("labels")]
            inputs = {k: v.to(device) for k, v in batch.items()}
            inputs["labels"] = labels
            
            outputs = model(**inputs)
            loss = outputs.loss

            if not torch.isfinite(loss):
                print(f"\n⚠️ Виявлено нескінченний loss ({loss.item()}). Пропускаємо цей крок.")
                continue

            if self.accumulation_steps > 1:
                loss = loss / self.accumulation_steps
            
            loss.backward()
            
            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(data_loader):
                optimizer.step()
                optimizer.zero_grad()
                display_loss = loss.item() * self.accumulation_steps if self.accumulation_steps > 1 else loss.item()
                writer.add_scalar('Train/Loss_step', display_loss, global_step)
                global_step += 1
            
            display_loss_pb = loss.item() * self.accumulation_steps if self.accumulation_steps > 1 else loss.item()
            progress_bar.set_postfix(loss=display_loss_pb)
        
        return global_step

    def _convert_box_cxcywh_to_xyxy(self, box, size):
        h, w = size.unbind(-1)
        center_x, center_y, box_w, box_h = box.unbind(-1)
        x_min = (center_x - 0.5 * box_w) * w
        y_min = (center_y - 0.5 * box_h) * h
        x_max = (center_x + 0.5 * box_w) * w
        y_max = (center_y + 0.5 * box_h) * h
        return torch.stack((x_min, y_min, x_max, y_max), dim=1)

    def _validate_one_epoch(self, model, data_loader, device):
        model.eval()
        metric = MeanAveragePrecision(box_format='xyxy').to(device)
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Validating")
            for batch in progress_bar:
                if batch is None: continue
                pixel_values = batch["pixel_values"].to(device)
                labels = [{k: v.to(device) for k, v in t.items()} for t in batch.pop("labels")]
                pixel_mask = batch.get("pixel_mask", None)
                if pixel_mask is not None: pixel_mask = pixel_mask.to(device)
                model_inputs = {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
                outputs = model(**model_inputs)
                orig_sizes = torch.stack([t["orig_size"] for t in labels]).to(device)
                results = self.image_processor.post_process_object_detection(
                    outputs, threshold=0.5, target_sizes=orig_sizes
                )
                targets_for_metric = []
                for i in range(len(labels)):
                    converted_boxes = self._convert_box_cxcywh_to_xyxy(labels[i]["boxes"], labels[i]["orig_size"])
                    targets_for_metric.append({
                        "boxes": converted_boxes, "labels": labels[i]["class_labels"]
                    })
                metric.update(results, targets_for_metric)
        try:
            mAP_dict = metric.compute()
            return mAP_dict.get('map', torch.tensor(0.0)).item()
        except Exception as e:
            print(f"Помилка при обчисленні mAP: {e}")
            return 0.0

    _check_for_resume_rcnn = FasterRCNNTrainer._check_for_resume_rcnn
    _save_checkpoint = FasterRCNNTrainer._save_checkpoint
    
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