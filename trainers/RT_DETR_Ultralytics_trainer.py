import os
import sys
import datetime as dt
import shutil 
from ultralytics import RTDETR
from trainers.trainers import BaseTrainer, log_dataset_statistics_to_tensorboard
import glob
import torch

class UltralyticsDatasetWrapper:
    """Адаптує датасет ultralytics до формату, який очікує наша функція статистики."""
    def __init__(self, ultralytics_dataset):
        self.dataset = ultralytics_dataset
        # Визначаємо розмір зображення для денормалізації
        imgsz = self.dataset.imgsz
        self.img_h, self.img_w = (imgsz, imgsz) if isinstance(imgsz, int) else (imgsz[0], imgsz[1])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        original_item = self.dataset[idx]
        bboxes_xywhn = original_item['bboxes']
        
        # Конвертуємо рамки з нормалізованого формату [x_center, y_center, w, h]
        # у піксельний формат [x_min, y_min, x_max, y_max]
        bboxes_xyxy = []
        for bbox in bboxes_xywhn:
            x_center, y_center, w, h = bbox
            x1 = (x_center - w / 2) * self.img_w
            y1 = (y_center - h / 2) * self.img_h
            x2 = (x_center + w / 2) * self.img_w
            y2 = (y_center + h / 2) * self.img_h
            bboxes_xyxy.append([x1, y1, x2, y2])
        
        # Повертаємо None для зображення та ціль у потрібному форматі
        return None, {'boxes': torch.tensor(bboxes_xyxy)}

def log_stats_callback(trainer):
    """
    Ця функція викликається автоматично бібліотекою ultralytics
    після підготовки даних, але до початку навчання.
    """
    print("\nВиклик колбеку для аналізу статистики датасету...")
    #wrapped_dataset = UltralyticsDatasetWrapper(trainer.train_loader.dataset)
    #log_dataset_statistics_to_tensorboard(wrapped_dataset, trainer.writer)


class RTDETRUltralyticsTrainer(BaseTrainer):
    """Керує процесом навчання RT-DETR через бібліотеку ultralytics."""

    def __init__(self, training_params, dataset_dir):
        super().__init__(training_params, dataset_dir)
        self.training_mode = None

    def _ask_training_mode(self):
        """Допоміжний метод, що запитує режим навчання."""
        print("\n   Оберіть режим навчання для RT-DETR:")
        print("     1: Fine-tuning (використовувати попередньо навчені ваги 'rtdetr-l.pt', швидше, рекомендовано)")
        print("     2: Full training (навчати модель з нуля, дуже довго)")
        while True:
            choice = input("   Ваш вибір режиму (1 або 2): ").strip()
            if choice == '1':
                print("✅ Ви обрали Fine-tuning.")
                return '_finetune'
            elif choice == '2':
                print("✅ Ви обрали Full training (з нуля).")
                return '_full'
            else:
                print("   ❌ Невірний вибір. Будь ласка, введіть 1 або 2.")

    def _select_training_mode(self):
        """Запитує у користувача режим навчання та повертає відповідний суфікс."""
        return self._ask_training_mode()

    def _get_model_name(self):
        """Повертає повну назву моделі для логування, враховуючи режим навчання."""
        if self.training_mode is None:
            return "RT-DETR (Ultralytics)"
        
        mode_name = "Fine-tune" if self.training_mode == '_finetune' else "Full Training"
        return f"RT-DETR (Ultralytics {mode_name})"

    def _check_for_resume(self, project_path):
        """
        Перевіряє наявність останнього незавершеного навчання в проєкті.
        Повертає шлях до чекпоінту та прапорець, чи потрібно відновлювати.
        """
        train_dirs = sorted(glob.glob(os.path.join(project_path, "train*")))
        if not train_dirs:
            return None, False
        
        last_train_dir = train_dirs[-1]
        last_model_path = os.path.join(last_train_dir, "weights", "last.pt")
        
        if os.path.exists(last_model_path):
            print(f"\n✅ Виявлено незавершене навчання: {last_train_dir}")
            answer = input("Бажаєте продовжити навчання з останньої точки збереження? (y/n): ").strip().lower()
            if answer in ['y', 'yes', 'н', 'так']:
                print(f"🚀 Навчання буде продовжено з файлу: {last_model_path}")
                return last_model_path, True
        
        print("🗑️ Попередній прогрес буде проігноровано. Навчання розпочнеться з нуля.")
        return None, False

    def start_or_resume_training(self, dataset_stats):
        if self.training_mode is None:
            self.training_mode = self._select_training_mode()
            
        print(f"\n--- Запуск тренування для {self._get_model_name()} ---")
        
        project_dir = os.path.join('runs', f'rtdetr_ultralytics{self.training_mode}')
        
        # 1. Перевірка на відновлення навчання
        resume_path, should_resume = self._check_for_resume(project_dir)
        
        model_to_load = None
        if should_resume:
            model_to_load = resume_path
            print(f"🔧 Модель для відновлення: {model_to_load}")
        else:
            if self.training_mode == '_finetune':
                model_to_load = 'rtdetr-l.pt'
                print("🔧 Створення моделі з попередньо навчених ваг 'rtdetr-l.pt'.")
            elif self.training_mode == '_full':
                model_to_load = 'rtdetr-l.yaml'
                print("🔧 Створення моделі з нуля за конфігурацією 'rtdetr-l.yaml'.")

        if model_to_load is None:
            print("❌ Не вдалося визначити, яку модель завантажувати.")
            sys.exit(1)

        # 2. Створення екземпляра моделі
        try:
            model = RTDETR(model_to_load)
        except Exception as e:
            print(f"❌ Помилка при завантаженні моделі з файлу '{model_to_load}': {e}")
            print("ℹ️ Переконайтеся, що файл 'rtdetr-l.pt' (для fine-tuning) або 'rtdetr-l.yaml' (для full training) існує та доступний.")
            return None
        
        model.add_callback("on_pretrain_routine_end", log_stats_callback)

        self.params['project'] = project_dir
        self.params['name'] = f'train_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'
            
        print("\n🚀 Розпочинаємо тренування моделі...")
        
        # 3. Тепер виклик функції чистий. Всі параметри передаються з одного місця.
        results = model.train(
            data='yolo_config.yaml',
            imgsz=(480, 640), #з практики, так ніби краще
            resume=should_resume,
            **self.params  # Розпаковуємо вже повністю налаштований словник
        )

        print("\n--- Результати тренування ---")
        print(f"Найкраща модель збережена у папці: {results.save_dir}")
        print(f"📈 Логи для TensorBoard можна знайти в тій самій директорії.")

        final_model_path = os.path.join(results.save_dir, "weights", "best.pt")
        final_path = None
        if os.path.exists(final_model_path):
            final_name_suffix = "Finetune" if self.training_mode == '_finetune' else "Full"
            final_path = f"Final-RTDETR-Ultralytics-{final_name_suffix}-best.pt"
            shutil.copy(final_model_path, final_path)
            print(f"\n✅ Найкращу модель скопійовано у файл: {final_path}")

        # Видаляємо змінені параметри з self.params, щоб вони не впливали на звіт
        clean_hyperparams = self.params.copy()
        clean_hyperparams.pop('project')
        clean_hyperparams.pop('name')

        summary = {
            "model_name": self._get_model_name(),
            "image_count": dataset_stats.get("image_count", "N/A"),
            "negative_count": dataset_stats.get("negative_count", "N/A"),
            "class_count": dataset_stats.get("class_count", "N/A"),
            "image_size": dataset_stats.get("image_size", "N/A"),
            "best_map": f"{results.results_dict.get('metrics/mAP50-95(B)', 0.0):.4f}",
            "best_model_path": final_path,
            "hyperparameters": clean_hyperparams
        }
        return summary