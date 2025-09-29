import os
import datetime as dt
import shutil 
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torchvision.models as models
from tqdm import tqdm
from trainers.trainers import BaseTrainer

class ResNetTrainer(BaseTrainer):
    """Керує процесом навчання моделі ResNet за допомогою PyTorch."""

    def start_or_resume_training(self, dataset_stats):
        imgsz = dataset_stats.get('image_size')
        print("\n--- Запуск тренування для ResNet ---")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔌 Обрано пристрій для навчання: {str(device).upper()}")

        if imgsz:
            print(f"🖼️ Розмір зображень для навчання буде змінено на {imgsz[0]}x{imgsz[1]}.")
        else:
            print("⚠️ Розмір зображення (imgsz) не передано, можлива помилка.")

        project_dir = self.params.get('project', 'runs/resnet')
        epochs = self.params.get('epochs', 25)
        batch_size = self.params.get('batch', 16)
        learning_rate = self.params.get('lr', 0.001)

        train_loader, val_loader, num_classes, class_names = self._prepare_dataloaders(imgsz, batch_size)
        print(f"📊 Знайдено {num_classes} класів: {class_names}")

        model = self._get_model(num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        run_name, checkpoint_path = self._check_for_resume_resnet(project_dir)
        start_epoch = 0
        best_accuracy = 0.0
        final_loss = 0.0

        if checkpoint_path:
            model, optimizer, start_epoch, best_accuracy = self._load_checkpoint(
                checkpoint_path, model, optimizer, device
            )
            print(f"🚀 Відновлення навчання з {start_epoch}-ї епохи.")
        
        run_dir = os.path.join(project_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        print(f"📂 Результати будуть збережені в: {run_dir}")

        print(f"\n🚀 Розпочинаємо тренування на {epochs} епох...")
        for epoch in range(start_epoch, epochs):
            train_loss, train_acc = self._train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
            val_loss, val_acc = self._validate_one_epoch(model, val_loader, criterion, device)

            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            is_best = val_acc > best_accuracy
            if is_best:
                best_accuracy = val_acc
                final_loss = val_loss

            self._save_checkpoint({
                'epoch': epoch + 1, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'best_accuracy': best_accuracy
            }, is_best, run_dir)

        print("\n🎉 Навчання успішно завершено!")
        
        best_model_path = os.path.join(run_dir, "best_model.pth")
        final_path = None
        if os.path.exists(best_model_path):
            final_path = "Final-ResNet-best.pth"
            shutil.copy(best_model_path, final_path)
            print(f"\n✅ Найкращу модель скопійовано у файл: {final_path} (Точність: {best_accuracy:.4f})")
        
        summary = {
            "model_name": self._get_model_name(),
            "image_count": dataset_stats.get("image_count", "N/A"),
            "negative_count": dataset_stats.get("negative_count", "N/A"),
            "class_count": dataset_stats.get("class_count", num_classes),
            "image_size": dataset_stats.get("image_size", "N/A"),
            "best_map": f"{best_accuracy:.4f} (Accuracy)",
            "final_loss": f"{final_loss:.4f}",
            "best_model_path": final_path,
            "hyperparameters": self.params
        }
        return summary


    def _prepare_dataloaders(self, imgsz, batch_size):
        """Готує завантажувачі даних для тренування та валідації."""
        img_height, img_width = (imgsz[1], imgsz[0]) if isinstance(imgsz, tuple) else (imgsz, imgsz)
        train_transforms = T.Compose([
            T.Resize((img_height, img_width)), T.RandomHorizontalFlip(),
            T.RandomRotation(10), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transforms = T.Compose([
            T.Resize((img_height, img_width)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = ImageFolder(root=os.path.join(self.dataset_dir, 'train'), transform=train_transforms)
        val_dataset = ImageFolder(root=os.path.join(self.dataset_dir, 'val'), transform=val_transforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        return train_loader, val_loader, len(train_dataset.classes), train_dataset.classes

    def _get_model(self, num_classes):
        """Завантажує попередньо навчену модель ResNet50 і адаптує її."""
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model

    def _check_for_resume_resnet(self, project_path):
        """Перевіряє наявність незавершеного навчання для ResNet."""
        train_dirs = sorted(glob(os.path.join(project_path, "train*")))
        run_name = f'train_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        if not train_dirs:
            return run_name, None
        last_train_dir = train_dirs[-1]
        last_model_path = os.path.join(last_train_dir, "last_checkpoint.pth")
        if os.path.exists(last_model_path):
            print(f"\n✅ Виявлено незавершене навчання: {last_train_dir}")
            answer = input("Бажаєте продовжити навчання з останньої точки збереження? (y/n): ").strip().lower()
            if answer in ['y', 'yes', 'так']:
                print(f"🚀 Навчання буде продовжено з файлу: {last_model_path}")
                return os.path.basename(last_train_dir), last_model_path
        print("🗑️ Попередній прогрес буде проігноровано. Навчання розпочнеться з нуля.")
        return run_name, None
        
    def _train_one_epoch(self, model, loader, criterion, optimizer, device, epoch_num):
        """Виконує одну епоху тренування."""
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch_num + 1} [Train]")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            progress_bar.set_postfix(loss=running_loss/total_samples, acc=correct_predictions.double()/total_samples)
        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions.double() / total_samples
        return epoch_loss, epoch_acc.item()

    def _validate_one_epoch(self, model, loader, criterion, device):
        """Виконує одну епоху валідації."""
        model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            progress_bar = tqdm(loader, desc="Validating")
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_predictions += torch.sum(preds == labels.data)
                total_samples += labels.size(0)
                progress_bar.set_postfix(loss=running_loss/total_samples, acc=correct_predictions.double()/total_samples)
        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions.double() / total_samples
        return epoch_loss, epoch_acc.item()
    
    def _save_checkpoint(self, state, is_best, run_dir):
        """Зберігає контрольну точку моделі."""
        last_path = os.path.join(run_dir, "last_checkpoint.pth")
        torch.save(state, last_path)
        if is_best:
            best_path = os.path.join(run_dir, "best_model.pth")
            shutil.copyfile(last_path, best_path)

    def _load_checkpoint(self, path, model, optimizer, device):
        """Завантажує контрольну точку."""
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        return model, optimizer, start_epoch, best_accuracy