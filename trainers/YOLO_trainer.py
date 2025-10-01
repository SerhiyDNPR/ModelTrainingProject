import os
import sys
import datetime as dt
import shutil 
from ultralytics import YOLO
from trainers.trainers import BaseTrainer, log_dataset_statistics_to_tensorboard
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

# --- Колбек-функція для запуску аналізу статистики ---
def log_stats_callback(trainer):
    """
    Ця функція викликається автоматично бібліотекою ultralytics
    після підготовки даних, але до початку навчання.
    """
    print("\nВиклик колбеку для аналізу статистики датасету...")
    
    # Перевіряємо, чи існує та ініціалізований логгер TensorBoard
    if hasattr(trainer, 'loggers') and hasattr(trainer.loggers, 'tb'):
        tensorboard_writer = trainer.loggers.tb
        wrapped_dataset = UltralyticsDatasetWrapper(trainer.train_loader.dataset)
        log_dataset_statistics_to_tensorboard(wrapped_dataset, tensorboard_writer)
        print("✅ Статистику датасету успішно записано в TensorBoard.")
    else:
        print("⚠️ TensorBoard логгер не знайдено або не активний. Пропускаємо запис статистики.")


class YOLOTrainer(BaseTrainer):
    """
    Керує процесом навчання моделей YOLO.
    На початку запитує у користувача, яку версію використовувати (v8 або v9)
    та в якому режимі її навчати (fine-tune або full training).
    """

    def __init__(self, training_params, dataset_dir):
        """
        Конструктор не приймає версію моделі, вона буде визначена пізніше.
        """
        super().__init__(training_params, dataset_dir)
        self.model_config = None # Буде заповнено після вибору користувача

    def _ask_training_mode(self):
        """
        Допоміжний метод, що запитує режим навчання у користувача.
        """
        print("\n   Оберіть режим навчання для YOLO:")
        print("     1: Fine-tuning (почати з ваг, навчених на COCO, швидше, рекомендовано)")
        print("     2: Full training (навчати 'з чистого аркуша', значно довше)")
        while True:
            sub_choice = input("   Ваш вибір режиму (1 або 2): ").strip()
            if sub_choice == '1':
                return 'finetune'
            elif sub_choice == '2':
                return 'full'
            else:
                print("   ❌ Невірний вибір. Будь ласка, введіть 1 або 2.")

    def _select_model_version(self):
        """
        Відображає меню вибору версії YOLO та режиму навчання,
        і повертає словник з конфігурацією.
        """
        print("\nБудь ласка, оберіть версію моделі YOLO для навчання:")
        print("  1: YOLOv8 (швидка, стабільна, гарний базовий варіант)")
        print("  2: YOLOv9 (новіша, потенційно точніша, може бути більш вимогливою)")

        model_choice = None
        while model_choice not in ['1', '2']:
            model_choice = input("Ваш вибір версії (1 або 2): ").strip()
            if model_choice not in ['1', '2']:
                print("❌ Невірний вибір. Будь ласка, введіть 1 або 2.")

        # Тепер запитуємо режим навчання
        training_mode = self._ask_training_mode()

        config = {}
        if model_choice == '1':
            base_name = 'YOLOv8'
            weights_prefix = 'yolov8n'
        else: # model_choice == '2'
            base_name = 'YOLOv9'
            weights_prefix = 'yolov9c'
        
        # Залежно від режиму, обираємо файл .pt (ваги) або .yaml (архітектура)
        if training_mode == 'finetune':
            config['name'] = f"{base_name} (Fine-tune)"
            config['weights'] = f"{weights_prefix}.pt"
            print(f"✅ Обрано донавчання (Fine-tuning). Модель почне з ваг '{config['weights']}'.")
        else: # training_mode == 'full'
            config['name'] = f"{base_name} (Full)"
            config['weights'] = f"{weights_prefix}.yaml"
            print(f"✅ Обрано повне навчання. Модель буде ініціалізована з нуля згідно з архітектурою '{config['weights']}'.")
            
        return config

    def _get_model_name(self):
        """Повертає назву моделі з конфігурації."""
        return self.model_config.get('name', 'YOLO') if self.model_config else 'YOLO'

    def start_or_resume_training(self, dataset_stats):
        if self.model_config is None:
            self.model_config = self._select_model_version()

        model_name_str = self._get_model_name()
        weights_file = self.model_config['weights']

        print(f"\n--- Запуск тренування для {model_name_str} ---")
        resume_path, should_resume = self._check_for_resume(self.params['project'])
        
        # model_to_load буде або шляхом до ваг для відновлення, 
        # або .pt для fine-tuning, або .yaml для повного навчання
        model_to_load = resume_path if should_resume else weights_file
        
        try:
            model = YOLO(model_to_load)
            if not should_resume:
                model.model.requires_grad_(True)
        except Exception as e:
            print(f"Помилка при завантаженні моделі {model_name_str}: {e}")
            return None
        
        model.add_callback("on_pretrain_routine_end", log_stats_callback)
            
        print("\n🚀 Розпочинаємо тренування моделі...")
        
        # Створюємо ім'я для папки запуску, яке є безпечним для файлової системи
        run_folder_name = f'{model_name_str.lower().replace(" ", "_").replace("(", "").replace(")", "")}_train_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'

        results = model.train(
            data='yolo_config.yaml',
            imgsz=dataset_stats['image_size'][0],
            name=run_folder_name,
            resume=should_resume,
            **self.params
        )

        print("\n--- Результати тренування ---")
        print(f"Найкраща модель збережена у папці: {results.save_dir}")
        print(f"📈 Логи для TensorBoard можна знайти в тій самій директорії.")

        final_model_path = os.path.join(results.save_dir, "weights", "best.pt")
        final_path = None
        if os.path.exists(final_model_path):
             # Динамічна назва файлу, що включає режим навчання
             final_file_name = f"Final-{model_name_str.replace(' ', '_').replace('(', '').replace(')', '')}-best.pt"
             shutil.copy(final_model_path, final_file_name)
             print(f"\n✅ Найкращу модель скопійовано у файл: {final_file_name}")
             final_path = final_file_name
        
        # Формуємо словник з результатами для логування
        summary = {
            "model_name": self._get_model_name(),
            "image_count": dataset_stats.get("image_count", "N/A"),
            "negative_count": dataset_stats.get("negative_count", "N/A"),
            "class_count": dataset_stats.get("class_count", "N/A"),
            "image_size": dataset_stats.get("image_size", "N/A"),
            "best_map": f"{results.results_dict.get('metrics/mAP50-95(B)', 0.0):.4f}",
            "best_model_path": final_path,
            "hyperparameters": self.params
        }
        return summary