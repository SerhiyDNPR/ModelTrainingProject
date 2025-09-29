# main.py

import cv2
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os
import torch
import datetime as dt
import json
from Validators.video_processor import VideoProcessor
from Validators.yolo_wrapper import YOLOWrapper
from Validators.yolov9_wrapper import YOLOv9Wrapper
from Validators.detr_wrapper import DETRWrapper
from Validators.deformable_detr_wrapper import DeformableDETRWrapper
from Validators.faster_rcnn_wrapper import FasterRCNNWrapper
from Validators.fcos_wrapper import FCOSWrapper
from Validators.rt_detr_ultralytics_wrapper import RTDETRUltralyticsWrapper
from Validators.mask_rcnn_wrapper import MaskRCNNWrapper
from Validators.retinanet_wrapper import RetinaNetWrapper


# --- ОСНОВНІ НАЛАШТУВАННЯ ---
VIDEO_DIR = r"C:\Users\serhi\OneDrive\CD_DSST\Article_syntetic_data\Data_for_tests\Video_interceptors"
FILENAME_FILTER = "Зала"
CLASS_NAMES = ['Zala', 'SuperCum']
CONF_THRESHOLD = 0.5

MODEL_FACTORIES = {
    "YOLOv8": YOLOWrapper,
    "YOLOv9": YOLOv9Wrapper,
    "DETR": DETRWrapper,
    "Deformable DETR": DeformableDETRWrapper,
    "Faster R-CNN (ResNet50/ResNet101/MobileNet)": FasterRCNNWrapper,
    "FCOS (ResNet50)": FCOSWrapper,
    "RT DETR (Ultralitics)": RTDETRUltralyticsWrapper,
    "Mask R-CNN (ResNet50)": MaskRCNNWrapper,
    "RetinaNet (ResNet50)": RetinaNetWrapper
}


def setup_gui(window_name, processor):
    """Створює вікно та трекбар."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Timeline", window_name, 0, 1000, processor.on_trackbar_change)

def load_filters_from_json():
    """Пропонує користувачу завантажити фільтри з файлу JSON."""
    root = tk.Tk()
    root.withdraw()
    
    filepath = filedialog.askopenfilename(
        title="Оберіть filters.json для завантаження (або скасуйте, щоб почати з нуля)",
        filetypes=[("JSON files", "*.json")]
    )
    
    if not filepath:
        print("ℹ️ Файл фільтрів не обрано. Створення нових фільтрів з нуля.")
        return []
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            filters_data = json.load(f)
        if isinstance(filters_data, list):
            return filters_data
        else:
            print("⚠️ Формат файлу фільтрів невірний (очікується список). Починаємо з нуля.")
            return []
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"❌ Помилка завантаження фільтрів: {e}. Починаємо з нуля.")
        return []

def main():
    root = tk.Tk()
    root.withdraw()

    try:
        num_models = int(input("Скільки моделей ви хочете порівняти? (введіть число): ").strip())
        if num_models < 1:
            num_models = 1
            print("Некоректне значення. Буде використано 1 модель.")
    except ValueError:
        num_models = 1
        print("Некоректний ввід. Буде використано 1 модель.")

    loaded_models = []
    model_choices = list(MODEL_FACTORIES.keys())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Обрано пристрій для обробки: {device.upper()}")

    for i in range(num_models):
        print(f"\n--- Налаштування Моделі №{i+1} ---")
        
        print("Будь ласка, оберіть тип моделі:")
        for j, name in enumerate(model_choices, 1):
            print(f"  {j}: {name}")
        
        try:
            choice_idx = int(input(f"Ваш вибір для моделі №{i+1}: ")) - 1
            if not 0 <= choice_idx < len(model_choices): raise ValueError
            model_name = model_choices[choice_idx]
            print(f"Ви обрали: {model_name}")
        except (ValueError, IndexError):
            print("Невірний вибір. Пропуск цієї моделі.")
            continue

        model_path = filedialog.askopenfilename(
            title=f"Оберіть файл ваг для {model_name} (.pth або .pt)",
            filetypes=[("Model files", "*.pth *.pt")]
        )
        if not model_path:
            print("Файл моделі не обрано. Пропуск цієї моделі.")
            continue
        print(f"Обрано файл ваг: {model_path}")

        # --- CHANGE: Added SAHI option logic ---
        use_tracker_for_this_model = False
        use_sahi_for_this_model = False

        if model_name == "YOLOv8":
            sahi_choice = input(f"Використовувати SAHI (slicing) для моделі '{model_name}'? (y/n): ").strip().lower()
            if sahi_choice in ['y', 'yes', 'н', 'так']:
                use_sahi_for_this_model = True
                use_tracker_for_this_model = False  # Force disable tracker
                print("✅ Для цієї моделі буде застосовано SAHI. ByteTrack вимкнено, оскільки вони несумісні.")
                break
        
        # Ask about tracker only if SAHI was not selected
        if not use_sahi_for_this_model:
            tracker_choice = input(f"Використовувати ByteTrack для моделі '{model_name}'? (y/n): ").strip().lower()
            if tracker_choice in ['y', 'yes', 'н', 'так']:
                use_tracker_for_this_model = True
                print("✅ Для цієї моделі буде застосовано ByteTrack.")
                break
            else:
                use_tracker_for_this_model = False
                print("☑️ Для цієї моделі буде застосовано звичайну детекцію.")

        try:
            model_wrapper_class = MODEL_FACTORIES[model_name]
            model_wrapper = model_wrapper_class(class_names=CLASS_NAMES, device=device)
            
            # Pass the use_sahi flag to the load method for YOLOv8 ---
            if model_name == "YOLOv8":
                model_wrapper.load(model_path, use_sahi=use_sahi_for_this_model)
            else:
                model_wrapper.load(model_path)
            
            filename_no_ext = os.path.splitext(os.path.basename(model_path))[0]
            
            loaded_models.append({
                'name': model_name,               
                'filename': filename_no_ext,      
                'wrapper': model_wrapper,
                'use_tracker': use_tracker_for_this_model 
            })
            
        except Exception as e:
            print(f"❌ Не вдалося завантажити модель {model_name} з файлу {model_path}. Помилка: {e}")

    if not loaded_models:
        print("Не завантажено жодної моделі. Вихід.")
        return

    initial_filters = load_filters_from_json()

    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_filename = f"Comparison_Results_{FILENAME_FILTER}_{timestamp}.csv"
    print(f"\n📝 Результати будуть записуватись у новий файл: {log_filename}")

    window_name = "Multi-Model Validation"
    
    processor = VideoProcessor(loaded_models, window_name, CONF_THRESHOLD, initial_filters=initial_filters)

    video_files = processor.find_video_files(VIDEO_DIR, FILENAME_FILTER)
    if not video_files: return
    
    setup_gui(window_name, processor)
    
    for video_path in video_files:
        print(f"\n▶️ Обробка відео: {video_path}")
        if processor.run_on_video(video_path):
            print("⏹️ Обробку перервано користувачем.")
            break

    if processor.log_records:
        final_df = pd.DataFrame(processor.log_records)
        final_df.to_csv(log_filename, index=False)
        print(f"\n✅ Обробку завершено. Результати збережено у файлі: {log_filename}")
    else:
        print("\nОбробку завершено. Немає даних для запису.")
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()