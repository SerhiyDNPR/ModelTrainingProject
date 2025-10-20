import os
import datetime as dt
import shutil 
from glob import glob
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models
from tqdm import tqdm
from DataSetUtils.PascalVOCDataset import PascalVOCDataset
from trainers.trainers import BaseTrainer, collate_fn
from torchmetrics.detection import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter

# --- Тренер для Mask R-CNN ---
class MaskRCNNTrainer(BaseTrainer):
    """Керує процесом навчання моделі Mask R-CNN (лише для детекції)."""

    def _get_model_name(self):
        return "MaskRCNN"

    def start_or_resume_training(self, dataset_stats):
        # Цей метод побудований за аналогією з FasterRCNNTrainer
        imgsz = dataset_stats.get('image_size')
        print(f"\n--- Запуск тренування для {self._get_model_name()} ---")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔌 Обрано пристрій для навчання: {str(device).upper()}")
        
        if imgsz:
            print(f"🖼️ Розмір зображень для навчання буде змінено на {imgsz[0]}x{imgsz[1]}.")

        project_dir = self.params.get('project', 'runs/mask-rcnn')
        epochs = self.params.get('epochs', 25)
        batch_size = self.params.get('batch', 4)
        learning_rate = self.params.get('lr', 0.0001)
        self.accumulation_steps = self.params.get('accumulation_steps', 1)

        if self.accumulation_steps > 1:
            effective_batch_size = batch_size * self.accumulation_steps
            print(f"🔄 Увімкнено накопичення градієнтів. Кроків: {self.accumulation_steps}. Ефективний batch_size: {effective_batch_size}")

        train_loader, val_loader, num_classes = self._prepare_dataloaders(batch_size)
        print(f"📊 Знайдено {num_classes - 1} класів (+1 фон). Всього класів для моделі: {num_classes}")

        model = self._get_model(num_classes).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Додаємо планувальник швидкості навчання
        step_size = self.params.get('lr_scheduler_step_size', 8)
        gamma = self.params.get('lr_scheduler_gamma', 0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        run_name, checkpoint_path = self._check_for_resume_rcnn(project_dir)
        start_epoch, best_map, global_step = 0, 0.0, 0
        
        run_dir = os.path.join(project_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard_logs'))
        print(f"📂 Результати будуть збережені в: {run_dir}")
        print(f"📈 Логи для TensorBoard будуть збережені в: {writer.log_dir}")
        
        if checkpoint_path:
            model, optimizer, start_epoch, best_map, lr_scheduler = self._load_checkpoint(
                checkpoint_path, model, optimizer, device, lr_scheduler
            )
            print(f"🚀 Відновлення навчання з {start_epoch}-ї епохи.")

        print(f"\n🚀 Розпочинаємо тренування на {epochs} епох...")
        for epoch in range(start_epoch, epochs):
            global_step = self._train_one_epoch(model, optimizer, train_loader, device, epoch, writer, global_step, imgsz)
            val_map = self._validate_one_epoch(model, val_loader, device, imgsz)
            
            lr_scheduler.step() # Крок планувальника

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
            final_path = f"Final-{self._get_model_name()}-best.pth"
            shutil.copy(best_model_path, final_path)
            print(f"\n✅ Найкращу модель скопійовано у файл: {final_path} (mAP: {best_map:.4f})")
        
        summary = {
            "model_name": self._get_model_name(),
            "image_count": dataset_stats.get("image_count", "N/A"),
            "negative_count": dataset_stats.get("negative_count", "N/A"),
            "class_count": dataset_stats.get("class_count", num_classes - 1),
            "image_size": dataset_stats.get("image_size", "N/A"),
            "best_map": f"{best_map:.4f}",
            "best_model_path": final_path,
            "hyperparameters": self.params
        }
        return summary

    def _prepare_dataloaders(self, batch_size):
        label_map_path = os.path.join(self.dataset_dir, 'label_map.txt')
        with open(label_map_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        label_map = {name: i+1 for i, name in enumerate(class_names)}
        num_classes = len(label_map) + 1
        train_dataset = PascalVOCDataset(os.path.join(self.dataset_dir, 'train'), transforms=None, label_map=label_map)
        val_dataset = PascalVOCDataset(os.path.join(self.dataset_dir, 'val'), transforms=None, label_map=label_map)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True)
        return train_loader, val_loader, num_classes
        
    def _get_model(self, num_classes):
        model = models.detection.maskrcnn_resnet50_fpn(weights=models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        return model        

    def _train_one_epoch(self, model, optimizer, data_loader, device, epoch, writer, global_step, imgsz):
        model.train()
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1} [Train]")
        optimizer.zero_grad()
        transforms_list = []
        if imgsz:
            height, width = imgsz[1], imgsz[0]
            transforms_list.append(T.Resize((height, width)))
        transforms_list.append(T.ToTensor())
        transforms = T.Compose(transforms_list)
        for i, (images_pil, targets_raw) in enumerate(progress_bar):
            images = [transforms(img).to(device) for img in images_pil]
            targets = []
            for i, t_raw in enumerate(targets_raw):
                target = {k: v.to(device) for k, v in t_raw.items()}
                num_objs = len(target['labels'])
                img_h, img_w = images[i].shape[-2:]
                target['masks'] = torch.zeros((num_objs, img_h, img_w), dtype=torch.uint8, device=device)
                targets.append(target)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
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

    def _validate_one_epoch(self, model, data_loader, device, imgsz):
        model.eval()
        metric = MeanAveragePrecision(box_format='xyxy').to(device)
        transforms_list = []
        if imgsz:
            height, width = imgsz[1], imgsz[0]
            transforms_list.append(T.Resize((height, width)))
        transforms_list.append(T.ToTensor())
        transforms = T.Compose(transforms_list)
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Validating")
            for images, targets in progress_bar:
                images = [transforms(img).to(device) for img in images]
                targets_for_metric = [{k: v.to(device) for k, v in t.items()} for t in targets]
                predictions = model(images)
                predictions_for_metric = [{'boxes': p['boxes'], 'scores': p['scores'], 'labels': p['labels']} for p in predictions]
                metric.update(predictions_for_metric, targets_for_metric)
        mAP_dict = metric.compute()
        return mAP_dict['map'].item()
    
    from trainers.FasterRCNNTrainer import FasterRCNNTrainer
    _check_for_resume_rcnn = FasterRCNNTrainer._check_for_resume_rcnn
    
    # Визначаємо власний _load_checkpoint, що підтримує lr_scheduler
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