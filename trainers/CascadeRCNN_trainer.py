import os
import sys
import datetime as dt
import shutil
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import stat

from DataSetUtils.PascalVOCDataset import PascalVOCDataset
from trainers.trainers import BaseTrainer, collate_fn, log_dataset_statistics_to_tensorboard
from torchmetrics.detection import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter

try:
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmengine.registry import MODELS
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample
    import mmengine
except ImportError:
    print("="*60)
    print("🔴 ПОМИЛКА: MMDetection або його залежності не встановлено!")
    print("   Будь ласка, встановіть їх, виконавши команди з інструкції,")
    print("   інакше навчання Cascade R-CNN буде неможливим.")
    print("="*60)
    sys.exit(1)

# Явно імпортуємо всі основні модулі, щоб гарантовано зареєструвати компоненти.
from mmdet.models.detectors import *
from mmdet.models.backbones import *
from mmdet.models.necks import *
from mmdet.models.roi_heads import *
from mmdet.models.dense_heads import *


class MMDetModelWrapper(nn.Module):
    """
    Клас-обгортка для моделі з MMDetection, що адаптує її для використання PyTorch/Torchvision.
    """
    def __init__(self, config_path, checkpoint_url, num_classes, backbone_type):
        super().__init__()
        
        # 1. Завантаження та налаштування конфігурації
        cfg = Config.fromfile(config_path)
        cfg.model.roi_head.bbox_head[0].num_classes = num_classes - 1
        cfg.model.roi_head.bbox_head[1].num_classes = num_classes - 1
        cfg.model.roi_head.bbox_head[2].num_classes = num_classes - 1
        
        # Явно встановлюємо "scope" на 'mmdet', щоб mmengine знав, де шукати моделі.
        mmengine.DefaultScope.get_instance('mmdet_scope', scope_name='mmdet')
        
        # 2. Створення моделі
        self.model = MODELS.build(cfg.model)

        # 3. Завантаження попередньо навчених ваг
        print(f"🔄 Завантаження ваг для '{backbone_type}' з MMDetection Model Zoo...")
        checkpoint = load_checkpoint(self.model, checkpoint_url, map_location='cpu', revise_keys=[(r'^roi_head\.bbox_head\.', '')])
        print("✅ Ваги успішно завантажено.")

    def forward(self, images, targets=None):
        """
        Універсальний forward, що працює і для навчання, і для валідації.
        """
        if isinstance(images, list):
            images = torch.stack(images, 0)
            
        img_shape = (images.shape[2], images.shape[3])

        if self.training and targets is not None:
            # --- РЕЖИМ НАВЧАННЯ ---
            data_samples = []
            for i, target in enumerate(targets):
                gt_instances = InstanceData()
                gt_instances.bboxes = target['boxes']
                gt_instances.labels = target['labels']
                
                metainfo = {
                    'img_id': i,
                    'img_shape': img_shape,
                    'ori_shape': img_shape,
                    'pad_shape': img_shape,
                    'scale_factor': (1.0, 1.0)
                }

                data_sample = DetDataSample(gt_instances=gt_instances, metainfo=metainfo)
                data_samples.append(data_sample)
            
            losses = self.model.loss(images, data_samples)
            return losses
        else:
            # --- РЕЖИМ ВАЛІДАЦІЇ ---
            data_samples = []
            for i in range(images.shape[0]):
                metainfo = {
                    'img_id': i,
                    'img_shape': img_shape,
                    'ori_shape': img_shape,
                    'pad_shape': img_shape,
                    'scale_factor': (1.0, 1.0)
                }
                data_samples.append(DetDataSample(metainfo=metainfo))
            
            predictions_list = self.model.predict(images, data_samples)
            
            results = []
            for pred_sample in predictions_list:
                results.append({
                    'boxes': pred_sample.pred_instances.bboxes,
                    'scores': pred_sample.pred_instances.scores,
                    'labels': pred_sample.pred_instances.labels,
                })
            return results

# Функція-обробник помилок для shutil.rmtree, яка знімає атрибут "тільки для читання"
def remove_readonly(func, path, _):
    """Знімає атрибут "тільки для читання" і повторює спробу видалення."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def get_cascade_rcnn_model_from_mmdet(backbone_type, num_classes):
    """
    Створює та повертає модель Cascade R-CNN з MMDetection.
    """
    if backbone_type == 'resnet50':
        config_path = 'configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py'
        checkpoint_url = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'
    elif backbone_type == 'resnet101':
        config_path = 'configs/cascade_rcnn/cascade-rcnn_r101_fpn_1x_coco.py'
        checkpoint_url = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth'
    else:
        raise ValueError(f"Непідтримуваний backbone '{backbone_type}' для Cascade R-CNN.")
    
    if not os.path.exists(config_path):
        print("📂 Конфігураційні файли не знайдено. Запуск завантаження з MMDetection...")
        temp_repo_dir = 'mmdetection_temp'
        original_cwd = os.getcwd()
        
        if os.path.exists(temp_repo_dir):
            shutil.rmtree(temp_repo_dir, onerror=remove_readonly)

        try:
            os.system(f'git clone --filter=blob:none --no-checkout https://github.com/open-mmlab/mmdetection.git {temp_repo_dir}')
            os.chdir(temp_repo_dir)
            os.system('git sparse-checkout init --cone')
            os.system('git sparse-checkout set configs')
            os.system('git checkout')
            os.chdir(original_cwd)
            shutil.move(os.path.join(temp_repo_dir, 'configs'), 'configs')
            print("✅ Конфігураційні файли успішно завантажено.")
        except Exception as e:
            os.chdir(original_cwd)
            print(f"❌ Не вдалося завантажити конфіги. Помилка: {e}")
            print("   Будь ласка, завантажте їх вручну з репозиторію MMDetection.")
            sys.exit(1)
        finally:
            if os.path.exists(temp_repo_dir):
                shutil.rmtree(temp_repo_dir, onerror=remove_readonly)
            
    return MMDetModelWrapper(config_path, checkpoint_url, num_classes, backbone_type)


class CascadeRCNNTrainer(BaseTrainer):
    """
    Керує процесом навчання моделі Cascade R-CNN, інтегрованої з MMDetection.
    """
    def __init__(self, training_params, dataset_dir):
        super().__init__(training_params, dataset_dir)
        self.backbone_type = None

    def _ask_training_mode(self):
        print("\n   Оберіть режим навчання:")
        print("     1: Fine-tuning (навчати тільки 'голови' каскаду, швидше, рекомендовано)")
        print("     2: Full training (навчати всю модель, довше)")
        while True:
            sub_choice = input("   Ваш вибір режиму (1 або 2): ").strip()
            if sub_choice == '1': return '_finetune'
            elif sub_choice == '2': return '_full'
            else: print("   ❌ Невірний вибір. Будь ласка, введіть 1 або 2.")

    def _select_backbone(self):
        print("\nБудь ласка, оберіть 'хребет' (backbone) для Cascade R-CNN:")
        print("  1: ResNet-50 (збалансований варіант)")
        print("  2: ResNet-101 (повільніший, потенційно точніший)")
        while True:
            choice = input("Ваш вибір (1 або 2): ").strip()
            backbone_base = None
            if choice == '1':
                print("✅ Ви обрали ResNet-50."); backbone_base = 'resnet50'
            elif choice == '2':
                print("✅ Ви обрали ResNet-101."); backbone_base = 'resnet101'
            else:
                print("❌ Невірний вибір. Будь ласка, введіть 1 або 2."); continue
            training_mode_suffix = self._ask_training_mode()
            return f"{backbone_base}{training_mode_suffix}"

    def _get_model_name(self):
        if self.backbone_type is None: return "Cascade R-CNN (Unknown)"
        parts = self.backbone_type.split('_')
        base_name_map = {'resnet50': 'ResNet50', 'resnet101': 'ResNet101'}
        mode_name_map = {'finetune': 'Fine-tune', 'full': 'Full'}
        base_name = base_name_map.get(parts[0], 'Unknown')
        mode_name = mode_name_map.get(parts[1], 'Training')
        return f"Cascade R-CNN ({base_name} {mode_name})"

    def _get_model(self, num_classes):
        print(f"🔧 Створення моделі: {self._get_model_name()}")
        backbone_name = self.backbone_type.split('_')[0]
        
        model_wrapper = get_cascade_rcnn_model_from_mmdet(backbone_name, num_classes)
        
        is_finetune = self.backbone_type.endswith('_finetune')
        if is_finetune:
            print("❄️ Заморожування ваг backbone. Навчання тільки 'голів' (fine-tuning).")
            for param in model_wrapper.model.backbone.parameters():
                param.requires_grad = False
        else:
            print("🔥 Усі ваги моделі розморожено для повного навчання (full training).")
            for param in model_wrapper.model.parameters():
                param.requires_grad = True
        return model_wrapper

    def start_or_resume_training(self, dataset_stats):
        """Запускає або відновлює навчання."""
        if self.backbone_type is None:
            self.backbone_type = self._select_backbone()

        imgsz = dataset_stats.get('image_size')
        print(f"\n--- Запуск тренування для {self._get_model_name()} ---")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔌 Обрано пристрій для навчання: {str(device).upper()}")

        project_dir = os.path.join('runs', f'cascade-rcnn-{self.backbone_type}')
        epochs = self.params.get('epochs', 25)
        batch_size = self.params.get('batch', 4)
        learning_rate = self.params.get('lr', 0.0001)
        self.accumulation_steps = self.params.get('accumulation_steps', 1)
        lr_step_size = self.params.get('lr_scheduler_step_size', 8)
        lr_gamma = self.params.get('lr_scheduler_gamma', 0.1)

        train_loader, val_loader, num_classes = self._prepare_dataloaders(batch_size)
        print(f"📊 Знайдено {num_classes - 1} класів (+1 фон). Всього класів для моделі: {num_classes}")

        model = self._get_model(num_classes).to(device)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

        run_name, checkpoint_path = self._check_for_resume_rcnn(project_dir)
        start_epoch, best_map = 0, 0.0
        
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

        global_step = 0
        for epoch in range(start_epoch, epochs):
            global_step = self._train_one_epoch(model, optimizer, train_loader, device, epoch, writer, global_step, imgsz)
            val_map = self._validate_one_epoch(model, val_loader, device, imgsz)
            scheduler.step()
            print(f"Epoch {epoch + 1}/{epochs} | Validation mAP: {val_map:.4f}")
            writer.add_scalar('Validation/mAP', val_map, epoch)
            writer.add_scalar('LearningRate/Main', optimizer.param_groups[0]['lr'], epoch)
            is_best = val_map > best_map
            if is_best: best_map = val_map
            self._save_checkpoint({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
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
        
        summary = { "model_name": self._get_model_name(), "best_map": f"{best_map:.4f}", "best_model_path": final_path, "hyperparameters": self.params }
        return summary

    def _prepare_dataloaders(self, batch_size):
        label_map_path = os.path.join(self.dataset_dir, 'label_map.txt')
        with open(label_map_path, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines()]
        label_map = {name: i for i, name in enumerate(class_names)} 
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
        
        transforms = T.Compose([T.ToTensor()])
        if imgsz: transforms.transforms.insert(0, T.Resize((imgsz[1], imgsz[0])))
        
        for i, (images, targets) in enumerate(progress_bar):
            images_list = [transforms(img).to(device) for img in images]
            targets_list = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images_list, targets_list)
            
            # --- ВИРІШЕННЯ ПРОБЛЕМИ ---
            # MMDetection може повертати списки тензорів для деяких loss'ів.
            # Цей код "розгортає" їх в єдиний список перед підсумовуванням.
            loss_components = []
            for loss in loss_dict.values():
                if isinstance(loss, list):
                    loss_components.extend(loss) # Додаємо всі тензори зі списку
                else:
                    loss_components.append(loss) # Додаємо один тензор
            
            # Тепер підсумовуємо середні значення всіх окремих компонентів втрат.
            losses = sum(l.mean() for l in loss_components)
            # --------------------------
            
            if self.accumulation_steps > 1: losses = losses / self.accumulation_steps
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
        
        transforms = T.Compose([T.ToTensor()])
        if imgsz: transforms.transforms.insert(0, T.Resize((imgsz[1], imgsz[0])))

        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Validating")
            for images, targets in progress_bar:
                images_t = [transforms(img).to(device) for img in images]
                targets_for_metric = [{k: v.to(device) for k, v in t.items()} for t in targets]

                predictions = model(images_t)
                metric.update(predictions, targets_for_metric)
        
        mAP_dict = metric.compute()
        return mAP_dict['map'].item()

    def _check_for_resume_rcnn(self, project_path):
        train_dirs = sorted(glob(os.path.join(project_path, "train*")))
        if not train_dirs:
            return f'train_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}', None
        last_train_dir = train_dirs[-1]
        last_model_path = os.path.join(last_train_dir, "last_checkpoint.pth")
        if os.path.exists(last_model_path):
            print(f"\n✅ Виявлено незавершене навчання: {last_train_dir}")
            answer = input("Бажаєте продовжити навчання? (y/n): ").strip().lower()
            if answer in ['y', 'yes', 'так']:
                print(f"🚀 Продовження з файлу: {last_model_path}")
                return os.path.basename(last_train_dir), last_model_path
        print("🗑️ Попередній прогрес буде проігноровано.")
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