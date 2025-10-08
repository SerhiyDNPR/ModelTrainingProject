# trainers/SSDTrainer.py

import os
import sys
import datetime as dt
import shutil 
from glob import glob
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models
from tqdm import tqdm
from DataSetUtils.PascalVOCDataset import PascalVOCDataset
from trainers.trainers import BaseTrainer, collate_fn, log_dataset_statistics_to_tensorboard
from torchmetrics.detection import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter

class SSDTrainer(BaseTrainer):
    """
    Керує процесом навчання моделей SSD (SSD300) та SSDLite (SSD320).
    На початку запитує у користувача, який backbone та режим навчання використовувати.
    """

    def __init__(self, training_params, dataset_dir):
        # NEW: Додано для зберігання вибору користувача
        super().__init__(training_params, dataset_dir)
        self.model_config = None

    # NEW: Метод для вибору режиму навчання (аналогічно до FasterRCNN)
    def _ask_training_mode(self):
        """Допоміжний метод, що запитує режим навчання."""
        print("\n   Оберіть режим навчання:")
        print("     1: Fine-tuning (навчати тільки 'голову', швидше, рекомендовано)")
        print("     2: Full training (навчати всю модель, довше)")
        while True:
            sub_choice = input("   Ваш вибір режиму (1 або 2): ").strip()
            if sub_choice == '1':
                return '_finetune'
            elif sub_choice == '2':
                return '_full'
            else:
                print("   ❌ Невірний вибір. Будь ласка, введіть 1 або 2.")

    # NEW: Метод для вибору бекбону та режиму
    def _select_backbone_and_mode(self):
        """Відображає меню вибору backbone та режиму, і повертає комбінований рядок."""
        print("\nБудь ласка, оберіть 'хребет' (backbone) для SSD:")
        print("  1: VGG16 (класичний, точний, але повільний)")
        print("  2: MobileNetV3-Large (сучасний, дуже швидкий, для real-time)")
        
        while True:
            choice = input("Ваш вибір (1 або 2): ").strip()
            backbone_base = None
            if choice == '1':
                print("✅ Ви обрали VGG16.")
                backbone_base = 'vgg16'
            elif choice == '2':
                print("✅ Ви обрали MobileNetV3-Large (SSDLite).")
                backbone_base = 'mobilenet'
            else:
                print("❌ Невірний вибір. Будь ласка, введіть 1 або 2.")
                continue

            training_mode_suffix = self._ask_training_mode()
            self.model_config = f"{backbone_base}{training_mode_suffix}"
            return self.model_config
    
    def _get_model(self, num_classes):
        """Завантажує модель SSD з обраним backbone та режимом навчання."""
        print(f"🔧 Створення моделі: {self._get_model_name()}")
        
        is_finetune = self.model_config.endswith('_finetune')
        
        if self.model_config.startswith('vgg16'):
            model = models.detection.ssd300_vgg16(weights=models.detection.SSD300_VGG16_Weights.DEFAULT)
        elif self.model_config.startswith('mobilenet'):
            model = models.detection.ssdlite320_mobilenet_v3_large(
                weights=models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        else:
            print(f"❌ Помилка: невідомий тип конфігурації '{self.model_config}'.")
            sys.exit(1)

        in_channels = [
            layer[0].in_channels if isinstance(layer, torch.nn.Sequential) else layer.in_channels
            for layer in model.head.classification_head.module_list
        ]
        
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = models.detection.ssd.SSDClassificationHead(
            in_channels, num_anchors, num_classes)
            
        if is_finetune:
            print("❄️ Заморожування ваг backbone. Навчання тільки 'голови' (fine-tuning).")
            for param in model.backbone.parameters():
                param.requires_grad = False
        else:
            print("🔥 Усі ваги моделі розморожено для повного навчання (full training).")

        return model

    def start_or_resume_training(self, dataset_stats):
        """Головний метод, що запускає або відновлює процес навчання."""
        if self.model_config is None:
            self._select_backbone_and_mode()

        print(f"\n--- Запуск тренування для {self._get_model_name()} ---")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔌 Обрано пристрій для навчання: {str(device).upper()}")
        
        # NEW: Динамічний розмір зображення залежно від моделі
        if self.model_config.startswith('vgg16'):
            imgsz = (512, 512)
            project_dir = os.path.join('runs', 'ssd-vgg16')
        else: # mobilenet
            imgsz = (512, 512)
            project_dir = os.path.join('runs', 'ssdlite-mobilenet')
            
        print(f"🖼️ Розмір зображень для навчання буде змінено на {imgsz[0]}x{imgsz[1]} (вимога моделі).")

        epochs = self.params.get('epochs', 25)
        batch_size = self.params.get('batch', 4)
        learning_rate = self.params.get('lr', 1e-5) # Зменшено для кращої стабільності
        self.accumulation_steps = self.params.get('accumulation_steps', 1)
        lr_step_size = self.params.get('lr_scheduler_step_size', 8)
        lr_gamma = self.params.get('lr_scheduler_gamma', 0.1)

        if self.accumulation_steps > 1:
            print(f"🔄 Увімкнено накопичення градієнтів. Ефективний batch_size: {batch_size * self.accumulation_steps}")

        train_loader, val_loader, num_classes = self._prepare_dataloaders(batch_size)
        print(f"📊 Знайдено {num_classes - 1} класів (+1 фон). Всього класів для моделі: {num_classes}")

        model = self._get_model(num_classes).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
        
        run_name, checkpoint_path = self._check_for_resume(project_dir)
        start_epoch, best_map, global_step = 0, 0.0, 0
        
        run_dir = os.path.join(project_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard_logs'))
        print(f"📂 Результати будуть збережені в: {run_dir}")

        log_dataset_statistics_to_tensorboard(train_loader.dataset, writer)

        if checkpoint_path:
            model, optimizer, scheduler, start_epoch, best_map = self._load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, device
            )
            print(f"🚀 Відновлення навчання з {start_epoch}-ї епохи.")
        
        print(f"\n🚀 Розпочинаємо тренування на {epochs} епох...")
        for epoch in range(start_epoch, epochs):
            global_step = self._train_one_epoch(model, optimizer, train_loader, device, epoch, writer, global_step, imgsz)
            val_map = self._validate_one_epoch(model, val_loader, device, imgsz)
            
            scheduler.step()
            print(f"Epoch {epoch + 1}/{epochs} | Validation mAP: {val_map:.4f}")

            writer.add_scalar('Validation/mAP', val_map, epoch)
            writer.add_scalar('LearningRate/Main', optimizer.param_groups[0]['lr'], epoch)

            is_best = val_map > best_map
            if is_best:
                best_map = val_map

            self._save_checkpoint({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), 'best_map': best_map
            }, is_best, run_dir)

        writer.close()
        print("\n🎉 Навчання успішно завершено!")
        
        best_model_path = os.path.join(run_dir, "best_model.pth")
        final_path = None
        if os.path.exists(best_model_path):
            model_name_safe = self._get_model_name().replace(' ', '_').replace('(', '').replace(')', '')
            final_path = f"Final-{model_name_safe}-best.pth"
            shutil.copy(best_model_path, final_path)
            print(f"\n✅ Найкращу модель скопійовано у файл: {final_path} (mAP: {best_map:.4f})")
        
        summary = {
            "model_name": self._get_model_name(),
            "image_count": dataset_stats.get("image_count", "N/A"),
            "class_count": num_classes - 1, "image_size": f"{imgsz[0]}x{imgsz[1]}",
            "best_map": f"{best_map:.4f}", "best_model_path": final_path,
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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)
        return train_loader, val_loader, num_classes

    def _train_one_epoch(self, model, optimizer, data_loader, device, epoch, writer, global_step, imgsz):
        model.train()
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1} [Train]")
        optimizer.zero_grad()
        transforms = T.Compose([T.Resize(imgsz), T.ToTensor()])
        for i, (images, targets) in enumerate(progress_bar):
            images = [transforms(img).to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            if self.accumulation_steps > 1:
                losses = losses / self.accumulation_steps
            losses.backward()
            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(data_loader):
                optimizer.step()
                optimizer.zero_grad()
                writer.add_scalar('Train/Loss_step', losses.item(), global_step)
                global_step += 1
            progress_bar.set_postfix(loss=losses.item())
        return global_step

    def _validate_one_epoch(self, model, data_loader, device, imgsz):
        model.eval()
        metric = MeanAveragePrecision(box_format='xyxy').to(device)
        transforms = T.Compose([T.Resize(imgsz), T.ToTensor()])
        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc="Validating"):
                images = [transforms(img).to(device) for img in images]
                targets_for_metric = [{k: v.to(device) for k, v in t.items()} for t in targets]
                predictions = model(images)
                metric.update(predictions, targets_for_metric)
        mAP_dict = metric.compute()
        return mAP_dict['map'].item()

    def _check_for_resume(self, project_path):
        train_dirs = sorted(glob(os.path.join(project_path, "train*")))
        if not train_dirs:
            return f'train_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}', None
        last_train_dir = train_dirs[-1]
        last_model_path = os.path.join(last_train_dir, "last_checkpoint.pth")
        if os.path.exists(last_model_path):
            print(f"\n✅ Виявлено незавершене навчання: {last_train_dir}")
            answer = input("Бажаєте продовжити? (y/n): ").strip().lower()
            if answer in ['y', 'yes', 'н', 'так']:
                return os.path.basename(last_train_dir), last_model_path
        return f'train_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}', None
        
    def _save_checkpoint(self, state, is_best, run_dir):
        last_path = os.path.join(run_dir, "last_checkpoint.pth")
        torch.save(state, last_path)
        if is_best:
            best_path = os.path.join(run_dir, "best_model.pth")
            shutil.copyfile(last_path, best_path)

    def _load_checkpoint(self, path, model, optimizer, scheduler, device):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_map = checkpoint.get('best_map', 0.0)
        return model, optimizer, scheduler, start_epoch, best_map