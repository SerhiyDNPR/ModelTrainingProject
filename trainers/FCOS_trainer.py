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

# --- Допоміжний клас для трансформацій ---
class DetectionTransforms:
    def __init__(self, is_train=True, cat_id_map=None, imgsz=None):
        self.cat_id_map = cat_id_map
        transforms_list = []
        if imgsz:
            height, width = imgsz[1], imgsz[0]
            transforms_list.append(T.Resize((height, width)))
        transforms_list.append(T.ToTensor())
        self.transforms = T.Compose(transforms_list)

    def __call__(self, image, target):
        if not target:
            transformed_target = {
                'boxes': torch.empty((0, 4), dtype=torch.float32),
                'labels': torch.empty(0, dtype=torch.int64)
            }
            image = self.transforms(image)
            return image, transformed_target

        boxes = [ann['bbox'] for ann in target]
        boxes = torch.tensor(boxes, dtype=torch.float32)
        if boxes.numel() > 0:
            boxes[:, 2:] += boxes[:, :2]
        labels = torch.tensor([self.cat_id_map[ann['category_id']] for ann in target], dtype=torch.int64)
        transformed_target = { 'boxes': boxes, 'labels': labels }
        image = self.transforms(image)
        return image, transformed_target

# --- Тренер для FCOS ---
class FCOSTrainer(BaseTrainer):
    """Керує процесом навчання моделі FCOS."""

    def start_or_resume_training(self, dataset_stats):
        imgsz = dataset_stats.get('image_size')
        print("\n--- Запуск тренування для FCOS ---")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔌 Обрано пристрій для навчання: {str(device).upper()}")
        
        if imgsz:
            print(f"🖼️ Розмір зображень для навчання буде змінено на {imgsz[0]}x{imgsz[1]}.")
        else:
            print("⚠️ Розмір зображення (imgsz) не передано, буде використано оригінальний розмір.")

        project_dir = self.params.get('project', 'runs/fcos')
        epochs = self.params.get('epochs', 25)
        batch_size = self.params.get('batch', 8)
        learning_rate = self.params.get('lr', 0.0001)
        step_size = self.params.get('lr_scheduler_step_size', 8)
        gamma = self.params.get('lr_scheduler_gamma', 0.1)
        self.accumulation_steps = self.params.get('accumulation_steps', 1)

        train_loader, val_loader, num_classes = self._prepare_dataloaders(batch_size, imgsz)
        print(f"📊 Знайдено {num_classes} класів. Навчання моделі для їх розпізнавання.")

        model = self._get_model(num_classes).to(device)
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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
        return train_loader, val_loader, num_classes
        
    def _get_model(self, num_classes):
        """Завантажує модель FCOS і адаптує її голову."""
        model = models.detection.fcos_resnet50_fpn(weights=models.detection.FCOS_ResNet50_FPN_Weights.DEFAULT)
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
        
        for i, (images, targets) in enumerate(progress_bar):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            if not torch.isfinite(losses):
                print(f"⚠️ Виявлено нескінченний loss на епосі {epoch + 1}. Пропускаємо крок. Loss: {losses.item()}")
                continue
            
            # Логіка накопичення градієнтів може бути додана тут, якщо потрібно
            # losses = losses / self.accumulation_steps
            # losses.backward()
            # if (i + 1) % self.accumulation_steps == 0:
            #   optimizer.step()
            #   optimizer.zero_grad()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            writer.add_scalar('Train/Loss_step', losses.item(), global_step)
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