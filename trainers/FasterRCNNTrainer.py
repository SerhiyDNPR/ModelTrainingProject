import os
import sys
import datetime as dt
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
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from trainers.backbone_factory import create_fpn_backbone

# --- Універсальний тренер для Faster R-CNN з вибором backbone ---
class FasterRCNNTrainer(BaseTrainer):
    """
    Керує процесом навчання моделі Faster R-CNN.
    На початку запитує у користувача, який backbone використовувати та в якому режимі його навчати.
    """

    def __init__(self, training_params, dataset_dir):
        super().__init__(training_params, dataset_dir)
        self.backbone_type = None

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
    
    def _select_backbone(self):
        """
        Відображає меню вибору backbone та режиму, і повертає комбінований рядок.
        """
        print("\nБудь ласка, оберіть 'хребет' (backbone) для Faster R-CNN:")
        print("  1: ResNet-50 (збалансований варіант)")
        print("  2: ResNet-101 (повільніший, потенційно точніший)")
        print("  3: MobileNetV3-Large (дуже швидкий, для мобільних пристроїв)")
        # --- НОВІ SWIN ТРАНСФОРМЕРИ ---
        print("  4: Swin-T (Tiny - Трансформер, швидкий)")
        print("  5: Swin-S (Small - Трансформер, збалансований)")
        # -----------------------------
        
        while True:
            choice = input("Ваш вибір (1, 2, 3, 4 або 5): ").strip() # !!! Змінено опції
            backbone_base = None
            if choice == '1':
                print("✅ Ви обрали ResNet-50.")
                backbone_base = 'resnet50'
            elif choice == '2':
                print("✅ Ви обрали ResNet-101.")
                backbone_base = 'resnet101'
            elif choice == '3':
                print("✅ Ви обрали MobileNetV3-Large.")
                backbone_base = 'mobilenet'
            elif choice == '4':
                print("✅ Ви обрали Swin-T.")
                backbone_base = 'swin_t'
                # Перевірка наявності timm для трансформерів
                try:
                    import timm
                except ImportError:
                    print("❌ Помилка: бібліотека 'timm' не встановлена. Оберіть інший backbone.")
                    continue
            elif choice == '5':
                print("✅ Ви обрали Swin-S.")
                backbone_base = 'swin_s'
                try:
                    import timm
                except ImportError:
                    print("❌ Помилка: бібліотека 'timm' не встановлена. Оберіть інший backbone.")
                    continue
            else:
                print("❌ Невірний вибір. Будь ласка, введіть 1, 2, 3, 4 або 5.")
                continue

            training_mode_suffix = self._ask_training_mode()
            return f"{backbone_base}{training_mode_suffix}"

    def _get_model_name(self):
        """Повертає повну назву моделі для логування, розбираючи self.backbone_type."""
        if self.backbone_type is None:
            return "Faster R-CNN (Unknown)"
        
        parts = self.backbone_type.split('_')
        base_name_map = {
            'resnet50': 'ResNet50',
            'resnet101': 'ResNet101',
            'mobilenet': 'MobileNet',
            'swin': 'Swin', # Буде Swin-T або Swin-S
        }
        mode_name_map = {
            'finetune': 'Fine-tune',
            'full': 'Full'
        }
        
        base_name = base_name_map.get(parts[0], 'Unknown')
        
        # Обробка Swin-T та Swin-S
        if parts[0] == 'swin':
            base_name = 'Swin-' + parts[1].upper() # Swin-T або Swin-S

        mode_name = mode_name_map.get(parts[-1], 'Training') # -1 для 'finetune'/'full'
        
        return f"Faster R-CNN ({base_name} {mode_name})"

    def _get_model(self, num_classes):
        """Завантажує модель Faster R-CNN з обраним backbone та режимом навчання."""
        print(f"🔧 Створення моделі: {self._get_model_name()}")
        
        is_finetune = self.backbone_type.endswith('_finetune')
        model = None
        
        if self.backbone_type.startswith('resnet50'):
            model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        
        elif self.backbone_type.startswith('resnet101'):
            # Спеціальна обробка ResNet101
            try:
                from torchvision.models import ResNet101_Weights
                backbone = resnet_fpn_backbone('resnet101', weights=ResNet101_Weights.IMAGENET1K_V1)
            except (ImportError, AttributeError):
                print("⚠️ Попередження: не вдалося завантажити ваги за новим API. Використовується 'pretrained=True'.")
                backbone = resnet_fpn_backbone('resnet101', pretrained=True)
            model = models.detection.FasterRCNN(backbone, num_classes=num_classes)
            # Для ResNet101 голова вже замінена, і ми виходимо раніше
            return model
        
        elif self.backbone_type.startswith('mobilenet'):
            model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                weights=models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            )

        # --- ЛОГІКА ДЛЯ SWIN ---
        elif self.backbone_type.startswith('swin'):
            # Визначаємо назву Swin моделі для timm
            swin_map = {
                'swin_t': 'swin_tiny_patch4_window7_224',
                'swin_s': 'swin_small_patch4_window7_224'
            }
            backbone_key = swin_map.get(self.backbone_type.split('_')[0] + '_' + self.backbone_type.split('_')[1], None)

            if backbone_key:
                print(f"🔧 Створення Swin FPN backbone: {backbone_key}")
                # Створюємо FPN з використанням Swin Transformer як бекбону
                backbone = create_fpn_backbone(self.backbone_type, pretrained=True, input_img_size = self.image_size[0])
                model = models.detection.FasterRCNN(backbone, num_classes=num_classes)
            else:
                print(f"❌ Помилка: невідомий тип Swin backbone.")
                sys.exit(1)
        # -----------------------
        
        else:
            print(f"❌ Помилка: невідомий тип backbone '{self.backbone_type}'.")
            sys.exit(1)
        
        # --- Заморожування ваг (загальна логіка) ---
        if is_finetune:
            print("❄️ Заморожування ваг backbone. Навчання тільки 'голови' (fine-tuning).")
            # Заморожуємо параметри backbone, які є частиною моделі
            for param in model.backbone.parameters():
                param.requires_grad = False
        else:
            print("🔥 Усі ваги моделі розморожено для повного навчання (full training).")
            
        # Замінюємо голову (FastRCNNPredictor)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
                
        return model

    def start_or_resume_training(self, dataset_stats):
        if self.backbone_type is None:
            self.backbone_type = self._select_backbone()

        imgsz = dataset_stats.get('image_size')
        print(f"\n--- Запуск тренування для {self._get_model_name()} ---")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔌 Обрано пристрій для навчання: {str(device).upper()}")
        
        if imgsz:
            print(f"🖼️ Розмір зображень для навчання буде змінено на {imgsz[0]}x{imgsz[1]}.")
        else:
            print("⚠️ Розмір зображення (imgsz) не передано, буде використано оригінальний розмір.")

        project_dir = os.path.join('runs', f'faster-rcnn-{self.backbone_type}')
        epochs = self.params.get('epochs', 25)
        batch_size = self.params.get('batch', 4)
        learning_rate = self.params.get('lr', 0.0001)
        self.accumulation_steps = self.params.get('accumulation_steps', 1)
        lr_step_size = self.params.get('lr_scheduler_step_size', 10)
        lr_gamma = self.params.get('lr_scheduler_gamma', 0.1)

        if self.accumulation_steps > 1:
            effective_batch_size = batch_size * self.accumulation_steps
            print(f"🔄 Увімкнено накопичення градієнтів. Кроків: {self.accumulation_steps}. Ефективний batch_size: {effective_batch_size}")

        train_loader, val_loader, num_classes = self._prepare_dataloaders(batch_size)
        print(f"📊 Знайдено {num_classes - 1} класів (+1 фон). Всього класів для моделі: {num_classes}")

        model = self._get_model(num_classes).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
        
        run_name, checkpoint_path = self._check_for_resume_rcnn(project_dir)
        start_epoch, best_map, global_step = 0, 0.0, 0
        
        run_dir = os.path.join(project_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard_logs'))
        print(f"📂 Результати будуть збережені в: {run_dir}")
        print(f"📈 Логи для TensorBoard будуть збережені в: {writer.log_dir}")

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

            self.save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_map': best_map
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
        num_classes = len(label_map) + 1 # N класів + 1 фон

        train_dataset = PascalVOCDataset(os.path.join(self.dataset_dir, 'train'), transforms=None, label_map=label_map)
        val_dataset = PascalVOCDataset(os.path.join(self.dataset_dir, 'val'), transforms=None, label_map=label_map)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

        return train_loader, val_loader, num_classes

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
                metric.update(predictions, targets_for_metric)
        
        mAP_dict = metric.compute()
        return mAP_dict['map'].item()

    def _check_for_resume_rcnn(self, project_path):
        train_dirs = sorted(glob(os.path.join(project_path, "train*")))
        if not train_dirs:
            run_name = f'train_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'
            return run_name, None
        
        last_train_dir = train_dirs[-1]
        last_model_path = os.path.join(last_train_dir, "last_checkpoint.pth")
        
        if os.path.exists(last_model_path):
            print(f"\n✅ Виявлено незавершене навчання: {last_train_dir}")
            answer = input("Бажаєте продовжити навчання з останньої точки збереження? (y/n): ").strip().lower()
            if answer in ['y', 'Y', 'н', 'Н']:
                print(f"🚀 Навчання буде продовжено з файлу: {last_model_path}")
                return os.path.basename(last_train_dir), last_model_path
        
        print("🗑️ Попередній прогрес буде проігноровано. Навчання розпочнеться з нуля.")
        run_name = f'train_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        return run_name, None
        
    def _load_checkpoint(self, path, model, optimizer, scheduler, device):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_map = checkpoint.get('best_map', 0.0)
        return model, optimizer, scheduler, start_epoch, best_map
