import os
import shutil
import json
from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PIL import Image
from collections import defaultdict # <-- ДОДАНО
from converters.converters import BaseDataConverter, remove_readonly

# --- Конвертер для Faster R-CNN (у форматі PASCAL VOC XML) ---
class PascalVOCDataConverter(BaseDataConverter):
    """Конвертує дані з Unity Perception у формат PASCAL VOC (XML) для Faster R-CNN."""

    def prepare_data(self):
        """Головний метод, що запускає процес конвертації для Faster R-CNN."""
        print("\n--- Розпочато конвертацію даних у формат Faster R-CNN (PASCAL VOC) ---")

        if self.output_dir.exists():
            print(f"🧹 Очищення існуючої директорії: {self.output_dir}")
            shutil.rmtree(self.output_dir, onerror=remove_readonly)

        source_dirs = sorted([p for p in self.source_dir.glob("solo*") if p.is_dir()], key=self._natural_sort_key)
        if not source_dirs:
            print(f"ПОМИЛКА: Не знайдено жодної директорії 'solo*' за шляхом '{self.source_dir}'")
            return

        # Створення базової структури папок
        for subset in ["train", "val", "test"]:
            (self.output_dir / subset).mkdir(parents=True, exist_ok=True)

        annotated_dirs = source_dirs
        negative_dir = None

        # 1. Виявлення класів (можна перевикористати логіку з YOLO)
        class_names = self._discover_classes(annotated_dirs)
        self._create_label_map(class_names) # Створюємо файл з мапою класів

        # 2. Збір всіх прикладів (позитивних та негативних)
        positive_examples, imgsz = self._gather_annotated_examples(annotated_dirs)
        negative_examples = self._gather_negative_examples(negative_dir)

        # 3. Розподіл даних
        print("\n🔄 Розподіл даних за вибірками (train/val/test)...")

        # 1. Розподіляємо позитивні приклади (з об'єктами) на всі 3 вибірки
        train_pos, test_pos = train_test_split(positive_examples, test_size=0.2, random_state=42)
        train_pos, val_pos = train_test_split(train_pos, test_size=0.125, random_state=42) # 0.125 * 0.8 = 0.1

        # 2. Розподіляємо негативні приклади (фон) на всі 3 вибірки
        if negative_examples:
            train_neg, test_neg = train_test_split(negative_examples, test_size=0.2, random_state=42)
            train_neg, val_neg = train_test_split(train_neg, test_size=0.125, random_state=42)
        else:
            train_neg, val_neg, test_neg = [], [], []
        
        # 3. Формуємо фінальні вибірки, додаючи негативні приклади також до тренувальної
        train_files = train_pos + train_neg
        val_files = val_pos + val_neg
        test_files = test_pos + test_neg

        splits = {"train": train_files, "val": val_files, "test": test_files}

        # 4. Створення структури папок та копіювання файлів з генерацією XML
        self._create_voc_structure(splits)

        # Статистика
        print("\n--- ✅ Статистика після конвертації ---")
        print(f"Тренувальна вибірка: {len(train_files)} зображень ({len(train_pos)} з об'єктами, {len(train_neg)} фонових)")
        print(f"Валідаційна вибірка: {len(val_files)} зображень ({len(val_pos)} з об'єктами, {len(val_neg)} фонових)")
        print(f"Тестова вибірка:    {len(test_files)} зображень ({len(test_pos)} з об'єктами, {len(test_neg)} фонових)")
        print("-----------------------------------------")
        total_images = len(train_files) + len(val_files) + len(test_files)
        print(f"🎉 Загальна кількість зображень: {total_images}")

        # Додавання "складних негативів"
        self._add_hard_negatives_rcnn()

        print(f"\n🎉 Конвертація даних для Faster R-CNN успішно завершена!")
        
        stats = {
            "image_size": imgsz,
            "image_count": total_images,
            "negative_count": len(negative_examples),
            "class_count": len(class_names)
        }
        return stats

    def _gather_annotated_examples(self, annotated_dirs):
        """Збирає інформацію про анотовані зображення."""
        positive_examples = []
        imgsz = None
        skipped_files = []
        print("\n🔎 Збір та аналіз файлів з анотаціями...")
        for directory in tqdm(annotated_dirs, desc="Аналіз позитивних прикладів", unit="папка"):
            json_files = [p.parent / "step0.frame_data.json" for p in directory.glob("sequence.*/step0.camera.png") if (p.parent / "step0.frame_data.json").exists()]
            for json_path in json_files:
                img_path = json_path.parent / "step0.camera.png"
                with open(json_path) as f:
                    frame_data = json.load(f)

                capture = frame_data.get("capture") or frame_data.get("captures", [{}])[0]
                if imgsz is None and capture.get("dimension"):
                    img_w, img_h = capture["dimension"]
                    imgsz = (int(img_w), int(img_h))

                voc_annotations = []
                annotations_list = frame_data.get("annotations", capture.get("annotations", []))
                for annotation in annotations_list:
                    if "BoundingBox2DAnnotation" in annotation.get("@type", ""):
                        for value in annotation.get("values", []):
                            class_name = value.get("label_name") or value.get("labelName")
                            if not class_name: continue

                            px_x, px_y = value["origin"]
                            px_w, px_h = value["dimension"]
                            # Конвертуємо у формат [xmin, ymin, xmax, ymax]
                            box = [int(px_x), int(px_y), int(px_x + px_w), int(px_y + px_h)]
                            voc_annotations.append({"class_name": class_name, "box": box})

                if voc_annotations:
                    positive_examples.append({"img_path": img_path, "img_size": imgsz, "annotations": voc_annotations})
                else:
                    skipped_files.append(img_path)

        print(f"\nЗнайдено {len(positive_examples)} позитивних прикладів з анотаціями.")
        if skipped_files:
            print(f"⚠️  Пропущено {len(skipped_files)} файлів через відсутність анотацій:")
            for file_path in skipped_files:
                print(f"   - {file_path}")
        return positive_examples, imgsz

    def _gather_negative_examples(self, negative_dir):
        """Збирає інформацію про негативні приклади."""
        negative_examples = []
        if negative_dir:
            print("🔎 Збір файлів з негативними прикладами...")
            all_negative_files = [p for p in negative_dir.glob("sequence.*/step0.camera.png")]
            for img_path in tqdm(all_negative_files, desc="Аналіз негативних прикладів"):
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                    current_img_size = (width, height)
                except Exception as e:
                    print(f"⚠️  Попередження: не вдалося прочитати розмір зображення {img_path}. Помилка: {e}. Цей файл буде пропущено.")
                    continue
                
                negative_examples.append({"img_path": img_path, "img_size": current_img_size, "annotations": []})
            print(f"Знайдено {len(negative_examples)} негативних прикладів.")
        return negative_examples

    def _create_voc_structure(self, splits):
        """Створює структуру папок та генерує XML для кожного зображення."""
        print("\n📦 Формування фінального датасету в форматі PASCAL VOC...")
        for split_name, files in splits.items():
            split_dir = self.output_dir / split_name
            for item in tqdm(files, desc=f"Обробка '{split_name}'", unit="file"):
                img_path = item['img_path']
                # Створюємо унікальне ім'я файлу
                parent_folder_name = img_path.parent.parent.name
                sequence_folder_name = img_path.parent.name
                unique_base_name = f"{parent_folder_name}_{sequence_folder_name}"

                # Копіюємо зображення
                shutil.copy(img_path, split_dir / f"{unique_base_name}.png")

                # Генеруємо та зберігаємо XML анотацію
                xml_content = self._generate_xml_annotation(
                    folder=split_name,
                    filename=f"{unique_base_name}.png",
                    img_size=item['img_size'],
                    annotations=item['annotations']
                )
                with open(split_dir / f"{unique_base_name}.xml", "w", encoding='utf-8') as f:
                    f.write(xml_content)

    def _generate_xml_annotation(self, folder, filename, img_size, annotations):
        """Генерує вміст XML-файлу у форматі PASCAL VOC."""
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = folder
        ET.SubElement(root, "filename").text = filename
        ET.SubElement(root, "path").text = "unknown" # Зазвичай не використовується

        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = "Unknown"

        if img_size:
            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(img_size[0])
            ET.SubElement(size, "height").text = str(img_size[1])
            ET.SubElement(size, "depth").text = "3"

        ET.SubElement(root, "segmented").text = "0"

        for ann in annotations:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = ann['class_name']
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(ann['box'][0])
            ET.SubElement(bndbox, "ymin").text = str(ann['box'][1])
            ET.SubElement(bndbox, "xmax").text = str(ann['box'][2])
            ET.SubElement(bndbox, "ymax").text = str(ann['box'][3])

        # Pretty-printing the XML
        rough_string = ET.tostring(root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def _discover_classes(self, annotated_dirs):
        """Сканує JSON-файли для виявлення унікальних класів та розрахунку статистики."""
        print("🔍 Сканування класів та збір статистики у вихідних даних Perception...")
        class_names = set()

        stats_dir = self.output_dir / "statistics"
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        for directory in tqdm(annotated_dirs, desc="Пошук класів", unit="папка"):
            dir_stats = defaultdict(list)

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
                                px_w, px_h = value["dimension"]
                                dir_stats[label_name].append((px_w, px_h))

            if dir_stats:
                stats_output = [f"--- Статистика для директорії '{directory.name}': ---"]
                for class_name, sizes in sorted(dir_stats.items()):
                    if sizes:
                        avg_w = sum(w for w, h in sizes) / len(sizes)
                        avg_h = sum(h for w, h in sizes) / len(sizes)
                        stats_output.append(f"     - Клас: '{class_name}', Середній розмір: {avg_w:.2f}x{avg_h:.2f} пікселів ({len(sizes)} об'єктів)")
                stats_output.append("   -------------------------------------------------")
                
                print("\n" + "\n".join(stats_output))

                stats_filename = stats_dir / f"stats_{directory.name}.txt"
                with open(stats_filename, 'w', encoding='utf-8') as f_stat:
                    f_stat.write("\n".join(stats_output))
                print(f"     💾 Статистику збережено у: {stats_filename}")
        
        sorted_names = sorted(list(class_names))
        print(f"\n✅ Всього знайдено {len(sorted_names)} унікальних класів: {sorted_names}")
        return sorted_names

    def _create_label_map(self, class_names):
        """Створює файл label_map.txt, потрібний для багатьох скриптів навчання."""
        print("✍️  Створення файлу 'label_map.txt' з іменами класів...")
        with open(self.output_dir / 'label_map.txt', 'w', encoding='utf-8') as f:
            for name in class_names:
                f.write(f"{name}\n")

    def _add_hard_negatives_rcnn(self):
        """Додає 'складні негативні' приклади з окремої папки до всіх вибірок."""
        answer = input("\nБажаєте додати Hard Negative приклади з окремої папки? (y/n): ").strip().lower()
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

        hn_images = glob(os.path.join(hard_negatives_dir, "*.jpg")) + glob(os.path.join(hard_negatives_dir, "*.png"))
        if not hn_images:
            print("⚠️ У вказаній директорії не знайдено зображень (.jpg або .png).")
            return
            
        # Розподіляємо складні негативи між train, val та test у пропорції ~70/10/20
        train_val_hn, test_hn = train_test_split(hn_images, test_size=0.2, random_state=42)
        train_hn, val_hn = train_test_split(train_val_hn, test_size=0.125, random_state=42)
        
        target_splits = {
            "train": train_hn,
            "val": val_hn,
            "test": test_hn
        }

        for split_name, files in target_splits.items():
            if not files: continue
            
            target_dir = self.output_dir / split_name
            print(f"\nКопіювання {len(files)} Hard Negative файлів до '{split_name}'...")
            for img_path_str in tqdm(files, desc=f"Hard Negatives to {split_name}", unit="file"):
                img_path = Path(img_path_str)
                base_name, _ = os.path.splitext(img_path.name)
                
                # Копіюємо зображення
                shutil.copy(img_path, target_dir / img_path.name)
                
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                    current_img_size = (width, height)
                except Exception as e:
                    print(f"⚠️  Попередження: не вдалося прочитати розмір зображення {img_path}. Помилка: {e}. XML для цього файлу не буде створено.")
                    continue

                # Створюємо порожній XML для негативного прикладу
                xml_content = self._generate_xml_annotation(
                    folder=split_name,
                    filename=img_path.name,
                    img_size=current_img_size,
                    annotations=[]
                )
                with open(target_dir / f"{base_name}.xml", "w", encoding='utf-8') as f:
                    f.write(xml_content)
                    
            print(f"✅ Успішно додано {len(files)} файлів до вибірки '{split_name}'.")

    def get_image_dimensions(self):
        """
        Швидко знаходить розміри першого знайденого зображення в вихідних даних.
        Це потрібно, якщо користувач пропускає етап конвертації.
        """
        print("🔍 Визначення розміру зображень з вихідних даних для PascalVOC...")
        source_dirs = [p for p in self.source_dir.glob("solo*") if p.is_dir()]
        if not source_dirs:
            print(f"ПОМИЛКА: Не знайдено папок 'solo*' в {self.source_dir}")
            return None

        for directory in source_dirs:
            try:
                # Шукаємо перше зображення .png у будь-якій підпапці sequence
                image_path = next(directory.glob("sequence.*/*.png"))
                with Image.open(image_path) as img:
                    width, height = img.size
                    print(f"✅ Розмір зображення визначено: {width}x{height}")
                    return (width, height)
            except (StopIteration, FileNotFoundError):
                # Продовжуємо, якщо в цій папці не знайдено зображень
                continue
        
        print("⚠️ Не вдалося визначити розмір зображення з вихідних файлів.")
        return None