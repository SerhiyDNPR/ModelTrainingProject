import datetime as dt
import os
import shutil
import torch
import cv2
import yaml
import json
import re
from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from multiprocessing import freeze_support
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog

# --- ФУНКЦІЇ, ПЕРЕНЕСЕНІ З КОНВЕРТЕРА ---

def natural_sort_key(s):
    """Ключ для природного сортування (solo_1, solo_2, solo_10)."""
    s = s.name
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def discover_classes(annotated_dirs):
    """Сканує всі JSON-файли для виявлення унікальних назв класів."""
    print("🔍 Сканування класів у вихідних даних Perception...")
    class_names = set()
    for directory in tqdm(annotated_dirs, desc="Пошук класів", unit="папка"):
        json_files = [p.parent / "step0.frame_data.json" for p in directory.glob("sequence.*/step0.camera.png") if (p.parent / "step0.frame_data.json").exists()]
        for frame_file in json_files:
            with open(frame_file) as f:
                frame_data = json.load(f)
            capture = frame_data.get("capture") or frame_data.get("captures", [{}])[0]
            annotations_list = frame_data.get("annotations", capture.get("annotations", []))
            for annotation in annotations_list:
                if "BoundingBox2DAnnotation" in annotation.get("@type", ""):
                    for value in annotation.get("values", []):
                        label_name = value.get("label_name") or value.get("labelName")
                        if label_name:
                            class_names.add(label_name)
    sorted_names = sorted(list(class_names))
    print(f"✅ Знайдено {len(sorted_names)} унікальних класів: {sorted_names}")
    return sorted_names

def convert_and_prepare_data(perception_source_dir, final_dataset_dir):
    """
    Конвертує дані з формату Perception напряму в фінальну структуру YoloDataset,
    розподіляючи на train, val, і test.
    """
    print("\n--- Розпочато процес конвертації та підготовки даних ---")
    
    if os.path.exists(final_dataset_dir):
        print(f"🧹 Очищення існуючої директорії: {final_dataset_dir}")
        shutil.rmtree(final_dataset_dir)

    base_input_path = Path(perception_source_dir)
    source_dirs = sorted([p for p in base_input_path.glob("solo*") if p.is_dir()], key=natural_sort_key)

    if not source_dirs:
        print(f"ПОМИЛКА: Не знайдено жодної директорії 'solo*' за шляхом '{base_input_path}'")
        return None, None

    images_dir = Path(final_dataset_dir) / "images"
    labels_dir = Path(final_dataset_dir) / "labels"
    for subset in ["train", "val", "test"]:
        os.makedirs(images_dir / subset, exist_ok=True)
        os.makedirs(labels_dir / subset, exist_ok=True)

    annotated_dirs = source_dirs[:-1] if len(source_dirs) > 1 else source_dirs
    negative_dir = source_dirs[-1] if len(source_dirs) > 1 else None

    class_names = discover_classes(annotated_dirs)
    class_map = {name: i for i, name in enumerate(class_names)}

    positive_examples = []
    negative_examples = []
    imgsz = (640, 480)

    print("\n🔎 Збір та аналіз файлів з анотаціями...")
    for directory in tqdm(annotated_dirs, desc="Аналіз позитивних прикладів", unit="папка"):
        all_files = sorted(
            [(p.parent / "step0.frame_data.json", p.parent / "step0.camera.png")
             for p in directory.glob("sequence.*/step0.camera.png") if (p.parent / "step0.frame_data.json").exists()],
            key=lambda x: int(x[1].parent.name.split('.')[-1])
        )
        
        print(f"   -> В папці '{directory.name}' знайдено {len(all_files)} файлів з розміткою.")

        for json_path, img_path in all_files:
            with open(json_path) as f:
                frame_data = json.load(f)
            
            capture = frame_data.get("capture") or frame_data.get("captures", [{}])[0]
            if capture.get("dimension"):
                img_w, img_h = capture["dimension"]
                imgsz = (int(img_w), int(img_h))
            
            yolo_annotations = []
            annotations_list = frame_data.get("annotations", capture.get("annotations", []))
            for annotation in annotations_list:
                if "BoundingBox2DAnnotation" in annotation.get("@type", ""):
                    for value in annotation.get("values", []):
                        class_name = value.get("label_name") or value.get("labelName")
                        class_id = class_map.get(class_name)
                        if class_id is None: continue
                        
                        px_x, px_y = value["origin"]
                        px_w, px_h = value["dimension"]
                        x_center_norm = (px_x + px_w / 2) / img_w
                        y_center_norm = (px_y + px_h / 2) / img_h
                        width_norm = px_w / img_w
                        height_norm = px_h / img_h
                        yolo_annotations.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")
            
            if yolo_annotations:
                positive_examples.append({"img_path": img_path, "annotations": yolo_annotations})

    print(f"\nЗнайдено {len(positive_examples)} позитивних прикладів з анотаціями.")

    if negative_dir:
        print("🔎 Збір файлів з негативними прикладами...")
        all_negative_files = [p for p in negative_dir.glob("sequence.*/step0.camera.png")]
        for img_path in tqdm(all_negative_files, desc="Аналіз негативних прикладів"):
            negative_examples.append({"img_path": img_path, "annotations": []})
        print(f"Знайдено {len(negative_examples)} негативних прикладів.")

    train_files, test_files = train_test_split(positive_examples, test_size=0.2, random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)
    
    train_files.extend(negative_examples)
    
    splits = {"train": train_files, "val": val_files, "test": test_files}

    print("\n📦 Формування фінального датасету...")
    for split_name, files in splits.items():
        img_dest_dir = images_dir / split_name
        lbl_dest_dir = labels_dir / split_name
        
        for item in tqdm(files, desc=f"Копіювання '{split_name}'", unit="file"):
            parent_folder_name = item['img_path'].parent.parent.name
            sequence_folder_name = item['img_path'].parent.name
            unique_base_name = f"{parent_folder_name}_{sequence_folder_name}"
            
            shutil.copy(item['img_path'], img_dest_dir / f"{unique_base_name}.png")

            label_filepath = lbl_dest_dir / f"{unique_base_name}.txt"
            with open(label_filepath, "w") as f_label:
                if item['annotations']:
                    f_label.write("\n".join(item['annotations']))
    
    # --- ДОДАНО: Виведення фінальної статистики ---
    total_train = len(train_files)
    total_val = len(val_files)
    total_test = len(test_files)
    total_images = total_train + total_val + total_test
    print("\n--- ✅ Статистика після конвертації ---")
    print(f"Тренувальна вибірка: {total_train} зображень")
    print(f"Валідаційна вибірка: {total_val} зображень")
    print(f"Тестова вибірка:    {total_test} зображень")
    print("-----------------------------------------")
    print(f"🎉 Загальна кількість зображень: {total_images}")
    # ----------------------------------------
    
    return imgsz, class_names

def check_for_unfinished_training():
    """Перевіряє наявність незавершеного навчання."""
    train_dirs = sorted(glob(os.path.join("runs", "detect", "train*")))
    if not train_dirs:
        return None
    last_train_dir = train_dirs[-1]
    last_model_path = os.path.join(last_train_dir, "weights", "last.pt")
    if os.path.exists(last_model_path):
        print(f"\n✅ Виявлено незавершене навчання: {last_train_dir}")
        answer = input("Бажаєте продовжити навчання з останньої точки збереження? (y/n): ").strip().lower()
        if answer not in ['y', 'yes', 'н', 'так']:
            print(f"🚀 Навчання буде продовжено з файлу: {last_model_path}")
            return last_model_path
        else:
            print("🗑️ Попередній прогрес буде проігноровано. Навчання розпочнеться з нуля.")
            return None
    return None

def add_hard_negatives(dataset_dir):
    """Додає 'складні негативні' приклади до тренувальної вибірки."""
    answer = input("\nБажаєте додати Hard Negative приклади до навчальної вибірки? (y/n): ").strip().lower()
    if answer not in ['y', 'yes', 'н', 'так']:
        print("Пропускаємо додавання Hard Negative прикладів.")
        return

    root = tk.Tk()
    root.withdraw()
    print("Будь ласка, оберіть директорію з Hard Negative прикладами...")
    hard_negatives_dir = filedialog.askdirectory(title="Оберіть директорію з Hard Negative прикладами")
    if not hard_negatives_dir:
        print("Директорію не обрано. Пропускаємо.")
        return

    train_images_dir = os.path.join(dataset_dir, "images", "train")
    train_labels_dir = os.path.join(dataset_dir, "labels", "train")
    hn_images = glob(os.path.join(hard_negatives_dir, "*.jpg")) + glob(os.path.join(hard_negatives_dir, "*.png"))
    if not hn_images:
        print("⚠️ У вказаній директорії не знайдено зображень (.jpg або .png).")
        return
        
    print(f"Копіювання {len(hn_images)} Hard Negative файлів...")
    for img_path in tqdm(hn_images, desc="Hard Negatives", unit="file"):
        shutil.copy(img_path, os.path.join(train_images_dir, os.path.basename(img_path)))
        base_name, _ = os.path.splitext(os.path.basename(img_path))
        open(os.path.join(train_labels_dir, f"{base_name}.txt"), 'w').close()
    print(f"✅ Успішно додано {len(hn_images)} файлів до тренувальної вибірки.")

if __name__ == '__main__':
    freeze_support()
    perception_source_dir = r"C:\Users\serhi\AppData\LocalLow\DefaultCompany\GenerateSynteticData"
    final_dataset_dir = "YoloDataset"

    do_conversion = input("Бажаєте запустити конвертацію даних з Unity Perception? (y/n): ").strip().lower()
    if do_conversion in ['y', 'yes', 'н', 'так']:
        if not os.path.isdir(perception_source_dir):
            print(f"ПОМИЛКА: Вказаний шлях до даних Perception '{perception_source_dir}' не існує.")
            exit(1)
            
        imgsz, class_names = convert_and_prepare_data(perception_source_dir, final_dataset_dir)
        if imgsz is None:
            print("Помилка під час підготовки даних. Роботу зупинено.")
            exit(1)

        add_hard_negatives(final_dataset_dir)

        print("\nСтворення файлу yolo_config.yaml для тренування...")
        with open("yolo_config.yaml", "w", encoding='utf-8') as f:
            f.write(f"path: {os.path.abspath(final_dataset_dir)}\n")
            f.write("train: images/train\nval: images/val\ntest: images/test\n\n")
            f.write(f"nc: {len(class_names)}\nnames: {class_names}\n")
    else:
        print("Конвертацію пропущено. Перевірка наявності існуючого датасету...")
        if not os.path.exists(final_dataset_dir) or not os.path.exists("yolo_config.yaml"):
             print(f"ERROR: Неможливо продовжити. Папка '{final_dataset_dir}' або файл 'yolo_config.yaml' відсутні.")
             exit(1)
        try:
            existing_image = glob(os.path.join(final_dataset_dir, "images", "train", "*.*"))[0]
            img = cv2.imread(existing_image)
            imgsz = (int(img.shape[1]), int(img.shape[0]))
        except IndexError:
            print("ERROR: Не знайдено зображень у тренувальній вибірці для визначення розміру.")
            exit(1)
    
    resume_path = check_for_unfinished_training()
    should_resume = resume_path is not None
    model_to_load = resume_path if should_resume else "yolov8n.pt"
    model = YOLO(model_to_load)

    print("\n🚀 Розпочинаємо тренування моделі...")
    model.train(
        data='yolo_config.yaml',
        epochs=40,
        imgsz=imgsz[0],
        batch=16,
        project='runs/detect',
        name=f'train_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}',
        exist_ok=False,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        patience=10,
        resume=should_resume
    )
    res_str = f"{imgsz[0]}x{imgsz[1]}"
    model.save(f"Final-YOLOv8n-{res_str}.pt")
    print(f"\n✅ Тренування завершено. Фінальну модель збережено як Final-YOLOv8n-{res_str}.pt")