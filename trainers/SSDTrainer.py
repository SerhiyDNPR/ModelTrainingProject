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
from torchvision.ops import Conv2dNormActivation

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
        super().__init__(training_params, dataset_dir)
        self.model_config = None

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

    def _get_model_name(self):
        """Повертає повну назву моделі для логування, базуючись на виборі користувача."""
        if not self.model_config: return "SSD (Unknown)"
        parts = self.model_config.split('_')
        base_name = "SSD (VGG16)" if parts[0] == 'vgg16' else "SSDLite (MobileNetV3)"
        mode_name = "Fine-tune" if parts[1] == 'finetune' else "Full"
        return f"{base_name} {mode_name}"
    
    def _get_model(self, num_classes):
        """Завантажує модель SSD з обраним backbone та режимом навчання."""
        print(f"🔧 Створення моделі: {self._get_model_name()}")
        
        is_finetune = self.model_config.endswith('_finetune')
        
        if self.model_config.startswith('vgg16'):
            model = models.detection.ssd300_vgg16(weights=models.detection.SSD300_VGG16_Weights.DEFAULT)
        elif self.model_config.startswith('mobilenet'):
            model = models.detection.ssdlite320_mobilenet_v3_large(weights=models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        else:
            sys.exit(f"❌ Помилка: невідомий тип конфігурації '{self.model_config}'.")

        in_channels = []
        for layer in model.head.classification_head.module_list:
            if isinstance(layer, torch.nn.Sequential) and isinstance(layer[0], Conv2dNormActivation):
                in_channels.append(layer[0][0].in_channels)
            else:
                in_channels.append(layer.in_channels)
        
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = models.detection.ssd.SSDClassificationHead(
            in_channels, num_anchors, num_classes)
            
        if is_finetune:
            print("❄️ Заморожування ваг backbone (fine-tuning).")
            for param in model.backbone.parameters():
                param.requires_grad = False
        else:
            print("🔥 Усі ваги моделі розморожено для повного навчання (full training).")

        return model

    def start_or_resume_training(self, dataset_stats):
        if self.model_config is None:
            self._select_backbone_and_mode()

        print(f"\n--- Запуск тренування для {self._get_model_name()} ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.model_config.startswith('vgg16'):
            imgsz = (512, 512)
            project_dir = os.path.join('runs', f'ssd-vgg16{self.model_config.split("vgg16")[-1]}')
        else: # mobilenet
            imgsz = (512, 512)
            project_dir = os.path.join('runs', f'ssdlite-mobilenet{self.model_config.split("mobilenet")[-1]}')
            
        print(f"🔌 Обрано пристрій: {str(device).upper()}. Розмір зображень: {imgsz[0]}x{imgsz[1]}.")

        epochs, batch_size, lr = self.params['epochs'], self.params['batch'], self.params['lr']
        self.accumulation_steps = self.params.get('accumulation_steps', 1)

        train_loader, val_loader, num_classes = self._prepare_dataloaders(batch_size)
        model = self._get_model(num_classes).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        lr_step_size = self.params.get('lr_scheduler_step_size', 8)
        lr_gamma = self.params.get('lr_scheduler_gamma', 0.1)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
        
        run_name, ckpt_path = self._check_for_resume(project_dir)
        start_epoch, best_map, global_step = self._load_checkpoint(ckpt_path, model, optimizer, scheduler, device)
        
        run_dir = os.path.join(project_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard_logs'))
        
        # <-- БЛОК TRY...FINALLY для гарантованого запису логів
        try:
            log_dataset_statistics_to_tensorboard(train_loader.dataset, writer)
            print(f"\n🚀 Розпочинаємо тренування на {epochs} епох...")
            
            for epoch in range(start_epoch, epochs):
                global_step = self._train_one_epoch(model, optimizer, train_loader, device, epoch, writer, global_step, imgsz)
                val_map = self._validate_one_epoch(model, val_loader, device, imgsz)
                scheduler.step()
                
                print(f"Epoch {epoch + 1}/{epochs} | Validation mAP: {val_map:.4f}")
                writer.add_scalar('Validation/mAP', val_map, epoch)
                writer.add_scalar('LearningRate/Main', optimizer.param_groups[0]['lr'], epoch)

                writer.flush()

                is_best = val_map > best_map
                if is_best: best_map = val_map
                self._save_checkpoint(epoch + 1, model, optimizer, scheduler, best_map, global_step, is_best, run_dir)

        finally:
            # <-- ДОДАНО: Гарантоване закриття writer, навіть при перериванні
            writer.close()
            print("\n🎉 Навчання завершено або перервано. Writer закрито.")


    def _prepare_dataloaders(self, batch_size):
        label_map_path = os.path.join(self.dataset_dir, 'label_map.txt')
        with open(label_map_path, 'r') as f: class_names = [line.strip() for line in f.readlines()]
        label_map = {name: i+1 for i, name in enumerate(class_names)}
        num_classes = len(label_map) + 1 
        train_dataset = PascalVOCDataset(os.path.join(self.dataset_dir, 'train'), transforms=None, label_map=label_map)
        val_dataset = PascalVOCDataset(os.path.join(self.dataset_dir, 'val'), transforms=None, label_map=label_map)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        return train_loader, val_loader, num_classes

    def _train_one_epoch(self, model, optimizer, data_loader, device, epoch, writer, global_step, imgsz):
        model.train()
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1} [Train]")
        transforms = T.Compose([T.Resize(imgsz), T.ToTensor()])
        optimizer.zero_grad()
        for i, (images, targets) in enumerate(progress_bar):
            images = [transforms(img).to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses = losses / self.accumulation_steps
            losses.backward()
            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(data_loader):
                optimizer.step()
                optimizer.zero_grad()
                writer.add_scalar('Train/Loss_step', losses.item() * self.accumulation_steps, global_step)
                global_step += 1
            progress_bar.set_postfix(loss=losses.item() * self.accumulation_steps)
        return global_step

    def _validate_one_epoch(self, model, data_loader, device, imgsz):
        model.eval()
        metric = MeanAveragePrecision(box_format='xyxy').to(device)
        transforms = T.Compose([T.Resize(imgsz), T.ToTensor()])
        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc="Validating"):
                images = [transforms(img).to(device) for img in images]
                predictions = model(images)
                metric.update(predictions, [{k: v.to(device) for k, v in t.items()} for t in targets])
        return metric.compute()['map'].item()

    def _check_for_resume(self, project_path):
        train_dirs = sorted(glob(os.path.join(project_path, "train*")))
        if not train_dirs: return f'train_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}', None
        last_dir = train_dirs[-1]
        last_ckpt = os.path.join(last_dir, "last_checkpoint.pth")
        if os.path.exists(last_ckpt):
            print(f"\n✅ Виявлено незавершене навчання: {last_dir}")
            if input("Бажаєте продовжити? (y/n): ").strip().lower() in ['y', 'yes', 'так', 'н']:
                return os.path.basename(last_dir), last_ckpt
        return f'train_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}', None
    
    def _load_checkpoint(self, path, model, optimizer, scheduler, device):
        if not path: return 0, 0.0, 0
        try:
            ckpt = torch.load(path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print(f"🚀 Відновлення навчання з {ckpt['epoch']}-ї епохи.")
            return ckpt['epoch'], ckpt.get('best_map', 0.0), ckpt.get('global_step', 0)
        except Exception as e:
            print(f"⚠️ Не вдалося завантажити чекпоінт: {e}. Починаємо з нуля.")
            return 0, 0.0, 0

    def _save_checkpoint(self, epoch, model, optimizer, scheduler, best_map, global_step, is_best, run_dir):
        state = {
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_map': best_map, 'global_step': global_step
        }
        last_path = os.path.join(run_dir, "last_checkpoint.pth")
        torch.save(state, last_path)
        if is_best:
            shutil.copyfile(last_path, os.path.join(run_dir, "best_model.pth"))