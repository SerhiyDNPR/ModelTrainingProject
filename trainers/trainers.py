# trainers.py

import os
import pprint
from abc import ABC, abstractmethod
from glob import glob
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch

class BaseTrainer(ABC):
    """Абстрактний базовий клас для всіх тренерів."""
    def __init__(self, training_params, dataset_dir):
        self.params = training_params
        self.dataset_dir = dataset_dir

    def display_hyperparameters(self):
        """Виводить гіперпараметри навчання у консоль у відформатованому вигляді."""
        print("\n--- Гіперпараметри навчання ---")
        pprint.pprint(self.params)
        print("---------------------------------")

    def _get_model_name(self):
        """
        Повертає назву моделі для логування, базуючись на назві класу.
        Може бути перевизначений у дочірніх класах для більш точної назви.
        """
        return self.__class__.__name__.replace("Trainer", "")

    @abstractmethod
    def start_or_resume_training(self, dataset_stats):
        """
        Запускає нове навчання або відновлює існуюче.
        
        Args:
            dataset_stats (dict): Словник зі статистикою про датасет.

        Returns:
            dict: Словник з результатами навчання для запису в лог-файл.
        """
        pass

    def _check_for_resume(self, project_path):
        """Перевіряє наявність незавершеного навчання."""
        train_dirs = sorted(glob(os.path.join(project_path, "*")))
        if not train_dirs:
            return None, False
        
        last_train_dir = train_dirs[-1]
        last_model_path = os.path.join(last_train_dir, "weights", "last.pt")

        if os.path.exists(last_model_path):
            print(f"\n✅ Виявлено незавершене навчання: {last_train_dir}")
            answer = input("Бажаєте продовжити навчання з останньої точки збереження? (y/n): ").strip().lower()
            if answer in ['y', 'Y', 'н', 'Н']:
                print(f"🚀 Навчання буде продовжено з файлу: {last_model_path}")
                return last_model_path, True
        
        print("🗑️ Попередній прогрес буде проігноровано. Навчання розпочнеться з нуля.")
        return None, False

def collate_fn(batch):
    """Спеціальна функція для об'єднання батчів у DataLoader."""
    return tuple(zip(*batch))

def log_dataset_statistics_to_tensorboard(dataset, writer: SummaryWriter):
    print("\n📊 Проводиться аналіз статистики тренувального датасету...")
    
    # 1. Збір статистики (без змін)
    bbox_widths, bbox_heights, bbox_areas, aspect_ratios = [], [], [], []
    for _, target in tqdm(dataset, desc="Аналіз датасету"):
        boxes = target.get('boxes')
        if boxes is None or len(boxes) == 0:
            continue
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes)
        widths_tensor = boxes[:, 2] - boxes[:, 0]
        heights_tensor = boxes[:, 3] - boxes[:, 1]
        for w, h in zip(widths_tensor, heights_tensor):
            w_item, h_item = w.item(), h.item()
            if w_item > 0 and h_item > 0:
                bbox_widths.append(w_item)
                bbox_heights.append(h_item)
                bbox_areas.append(w_item * h_item)
                aspect_ratios.append(w_item / h_item)

    if not bbox_widths:
        print("⚠️ Не знайдено жодної рамки в датасеті для аналізу.")
        return

    print("📈 Малювання 6 розподілів у вигляді лінійних графіків...")
    
    num_bins = 50 # Кількість точок на графіку

    # --- Графік 1: Лінія розподілу ширин ---
    counts, bin_edges = np.histogram(bbox_widths, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for i, count in enumerate(counts):
        writer.add_scalar('Dataset_Distribution_Lines/1_Widths', count, bin_centers[i])

    # --- Графік 2: Лінія розподілу ширин (логарифмічна шкала) ---
    log_widths = np.log1p(np.array(bbox_widths))
    counts, bin_edges = np.histogram(log_widths, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for i, count in enumerate(counts):
        writer.add_scalar('Dataset_Distribution_Lines/2_Widths_Log_Scale', count, bin_centers[i])

    # --- Графік 3: Лінія розподілу висот ---
    counts, bin_edges = np.histogram(bbox_heights, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for i, count in enumerate(counts):
        writer.add_scalar('Dataset_Distribution_Lines/3_Heights', count, bin_centers[i])

    # --- Графік 4: Лінія розподілу висот (логарифмічна шкала) ---
    log_heights = np.log1p(np.array(bbox_heights))
    counts, bin_edges = np.histogram(log_heights, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for i, count in enumerate(counts):
        writer.add_scalar('Dataset_Distribution_Lines/4_Heights_Log_Scale', count, bin_centers[i])

    # --- Графік 5: Лінія розподілу площ ---
    counts, bin_edges = np.histogram(bbox_areas, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for i, count in enumerate(counts):
        writer.add_scalar('Dataset_Distribution_Lines/5_Areas', count, bin_centers[i])

    # --- Графік 6: Лінія розподілу співвідношень сторін ---
    counts, bin_edges = np.histogram(aspect_ratios, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for i, count in enumerate(counts):
        writer.add_scalar('Dataset_Distribution_Lines/6_Aspect_Ratios', count, bin_centers[i])

    print("✅ Аналіз датасету успішно завершено.")
