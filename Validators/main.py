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
from Validators.ssd_wrapper import SSDWrapper
from Validators.efficientdet_wrapper import EfficientDetWrapper
from Validators.cascade_rcnn_wrapper import CascadeRCNNWrapper

# --- ОСНОВНІ НАЛАШТУВАННЯ ---
VIDEO_DIR = r"C:\Users\serhi\OneDrive\CD_DSST\Article_syntetic_data\Data_for_tests\Video_interceptors"
FILENAME_FILTER = "Зала"
#CLASS_NAMES = ['Zala', 'SuperCum']
CLASS_NAMES = ['Zala']
CONF_THRESHOLD = 0.5

MODEL_FACTORIES = {
    "YOLOv8": YOLOWrapper,
    "YOLOv9": YOLOv9Wrapper,
    "DETR": DETRWrapper,
    "Deformable DETR": DeformableDETRWrapper,
    "Faster R-CNN (ResNet50/ResNet101/MobileNet/Swin)": FasterRCNNWrapper,
    "Cascade R-CNN (ResNet50/ResNet101)": CascadeRCNNWrapper,
    "FCOS (ResNet50/EfficientNet(?))": FCOSWrapper,
    "RT DETR (Ultralitics)": RTDETRUltralyticsWrapper,
    "Mask R-CNN (ResNet50)": MaskRCNNWrapper,
    "RetinaNet (ResNet50/EfficientNet/Swin)": RetinaNetWrapper,
    "SSD (VGG16/MobileNetV3)": SSDWrapper,
    "EfficientDet": EfficientDetWrapper,
}

sahi_supported_models = ["YOLOv8", "SSD (VGG16/MobileNetV3)"] # <-- Перевірте, що назва тут ідентична

def setup_gui(window_name, processor):
    """Створює вікно та трекбар."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Timeline", window_name, 0, 1000, processor.on_trackbar_change)

def load_filters_from_json():
    """Пропонує користувачу завантажити фільтри з файлу JSON."""
    root = tk.Tk()
    root.withdraw()
    
    root.attributes('-topmost', True)
    
    filepath = filedialog.askopenfilename(
        title="Оберіть filters.json для завантаження (або скасуйте, щоб почати з нуля)",
        filetypes=[("JSON files", "*.json")]
    )
    
    root.attributes('-topmost', False)
    
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

    loaded_models = []
    model_choices = list(MODEL_FACTORIES.keys())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Обрано пристрій для обробки: {device.upper()}")

    while True:
        print("\n--- Оберіть тип моделей для завантаження ---")
        for j, name in enumerate(model_choices, 1):
            print(f"  {j}: {name}")
        
        try:
            choice_idx = int(input(f"Ваш вибір типу моделі: ")) - 1
            if not 0 <= choice_idx < len(model_choices):
                raise ValueError
            model_name = model_choices[choice_idx]
            print(f"✅ Ви обрали тип: {model_name}")
        except (ValueError, IndexError):
            print("❌ Невірний вибір. Спробуйте ще раз.")
            continue

        root.attributes('-topmost', True)

        model_paths = filedialog.askopenfilenames(
            title=f"Оберіть ОДИН або БІЛЬШЕ файлів ваг для '{model_name}' (.pth або .pt)",
            filetypes=[("Model files", "*.pth *.pt")]
        )
        
        root.attributes('-topmost', False)
        
        if not model_paths:
            print("⚠️ Файли моделей не обрано. Вибір типу скасовано.")
            add_more = input("\nБажаєте додати інший тип моделей? (y/n): ").strip().lower()
            if add_more in ['y', 'Y', 'н', 'Н']:
                continue
            else:
                break
                
        
        print(f"🔍 Обрано {len(model_paths)} файлів для типу '{model_name}'. Налаштуйте кожен з них.")

        for model_path in model_paths:
            print(f"\n--- Налаштування для файлу: {os.path.basename(model_path)} ---")

            use_tracker_for_this_model = False
            use_sahi_for_this_model = False

            if model_name in sahi_supported_models:
                sahi_choice = input(f"Використовувати SAHI (slicing) для цієї моделі? (y/n): ").strip().lower()
                if sahi_choice in ['y', 'Y', 'н', 'Н']:
                    use_sahi_for_this_model = True
                    use_tracker_for_this_model = False
                    print("✅ Для цієї моделі буде застосовано SAHI. ByteTrack вимкнено, оскільки вони несумісні.")
            
            if not use_sahi_for_this_model:
                tracker_choice = input(f"Використовувати ByteTrack для цієї моделі? (y/n): ").strip().lower()
                if tracker_choice in ['y', 'Y', 'н', 'Н']:
                    use_tracker_for_this_model = True
                    print("✅ Для цієї моделі буде застосовано ByteTrack.")
                else:
                    use_tracker_for_this_model = False
                    print("☑️ Для цієї моделі буде застосовано звичайну детекцію.")

            try:
                model_wrapper_class = MODEL_FACTORIES[model_name]
                model_wrapper = model_wrapper_class(class_names=CLASS_NAMES, device=device)
                
                if model_name in sahi_supported_models:
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

        add_more = input("\nБажаєте додати інший тип моделей? (y/n): ").strip().lower()
        if add_more not in ['y', 'yes', 'так']:
            break

    if not loaded_models:
        print("\nНе завантажено жодної моделі. Вихід.")
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