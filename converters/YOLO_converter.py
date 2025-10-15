import os
import shutil
import json
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from collections import defaultdict
from converters.converters import BaseDataConverter, remove_readonly

class YOLODataConverter(BaseDataConverter):
    """Конвертує дані з Unity Perception у формат YOLO."""
    def prepare_data(self):
        print("\n--- Розпочато конвертацію даних у формат YOLO ---")
        
        if self.output_dir.exists():
            print(f"🧹 Очищення існуючої директорії: {self.output_dir}")
            shutil.rmtree(self.output_dir, onerror=remove_readonly)

        source_dirs = sorted([p for p in self.source_dir.glob("solo*") if p.is_dir()], key=self._natural_sort_key)
        if not source_dirs:
            print(f"ПОМИЛКА: Не знайдено жодної директорії 'solo*' за шляхом '{self.source_dir}'")
            return

        images_dir = self.output_dir / "images"
        labels_dir = self.output_dir / "labels"
        for subset in ["train", "val", "test"]:
            (images_dir / subset).mkdir(parents=True, exist_ok=True)
            (labels_dir / subset).mkdir(parents=True, exist_ok=True)

        annotated_dirs = source_dirs[:-1] if len(source_dirs) > 1 else source_dirs
        negative_dir = source_dirs[-1] if len(source_dirs) > 1 else None
        
        print("🔍 Сканування класів у вихідних даних...")
        class_names = self._discover_classes(annotated_dirs)
        class_map = {name: i for i, name in enumerate(class_names)}
        print(f"✅ Знайдено {len(class_names)} унікальних класів: {class_names}")

        positive_examples = []
        negative_examples = []

        imgsz = self._copy_annotated_images(annotated_dirs, class_map, positive_examples)
        negative_count = self._copy_negative_examples(negative_dir, negative_examples)
        train_files, test_files, val_files = self._format_yolo_training_set(images_dir, labels_dir, positive_examples, negative_examples)
        
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

        self._add_hard_negatives()

        # ----------------------------------------
        print("\n📦 Обробка файлів та створення анотацій...")
        print("📊 Створення файлу конфігурації 'yolo_config.yaml'...")
        self._create_yaml_config(class_names)
        
        print("\n🎉 Конвертація даних для YOLO успішно завершена!")

        stats = {
            "image_size": imgsz,
            "image_count": total_images,
            "negative_count": negative_count,
            "class_count": len(class_names)
        }
        return stats

    def _copy_negative_examples(self, negative_dir, negative_examples):
        if not negative_dir:
            return 0
        print("🔎 Збір файлів з негативними прикладами...")
        all_negative_files = [p for p in negative_dir.glob("sequence.*/step0.camera.png")]
        for img_path in tqdm(all_negative_files, desc="Аналіз негативних прикладів"):
            negative_examples.append({"img_path": img_path, "annotations": []})
        print(f"Знайдено {len(negative_examples)} негативних прикладів.")
        return len(all_negative_files)

    def _format_yolo_training_set(self, images_dir, labels_dir, positive_examples, negative_examples):
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
        return train_files,test_files,val_files

    def _copy_annotated_images(self, annotated_dirs, class_map, positive_examples):
        imgsz = None
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
                    if imgsz is None:
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
        return imgsz

    def _discover_classes(self, annotated_dirs):
        """Сканує всі JSON-файли для виявлення унікальних назв класів та розрахунку статистики."""
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

    def _create_yaml_config(self, class_names):
        """Створює конфігураційний файл для YOLO."""
        config_data = {
            'path': str(self.output_dir.resolve()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(class_names),
            'names': class_names,
        }
        with open('yolo_config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, sort_keys=False, allow_unicode=True)

    def _add_hard_negatives(self):
        """Додаєmo 'складні негативні' приклади до тренувальної вибірки."""
        answer = input("\nБажаєте додати Hard Negative приклади до навчальної вибірки? (y/n): ").strip().lower()
        if answer not in ['y', 'Y', 'н', 'Н']:
            print("Пропускаємо додавання Hard Negative прикладів.")
            return

        root = tk.Tk()
        root.withdraw()
        print("Будь ласка, оберіть директорію з Hard Negative прикладами...")
        hard_negatives_dir = filedialog.askdirectory(title="Оберіть директорію з Hard Negative прикладами")
        if not hard_negatives_dir:
            print("Директорію не обрано. Пропускаємо.")
            return

        train_images_dir = os.path.join(self.output_dir, "images", "train")
        train_labels_dir = os.path.join(self.output_dir, "labels", "train")
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

    def get_image_dimensions(self):
        """
        Швидко знаходить розміри першого знайденого зображення в **конвертованій** директорії.
        Це використовується, коли конвертація пропускається, але потрібно знати розмір для тренування.
        """
        print(f"🔍 Визначення розміру зображень з раніше конвертованих даних у '{self.output_dir}'...")
        
        # Визначаємо потенційні директорії, де можуть бути зображення (train, val, test)
        search_dirs = [
            self.output_dir / "images" / "train",
            self.output_dir / "images" / "val",
            self.output_dir / "images" / "test"
        ]

        for directory in search_dirs:
            if not directory.exists():
                continue  # Пропускаємо, якщо директорії не існує

            try:
                # Шукаємо перше-ліпше зображення (.png або .jpg)
                image_path = next(directory.glob("*.[jp][pn]g"))
                with Image.open(image_path) as img:
                    width, height = img.size
                    print(f"✅ Розмір зображення визначено: {width}x{height} (з файлу {image_path.name})")
                    return (width, height)
            except StopIteration:
                # Продовжуємо пошук у наступній папці, якщо в поточній нічого не знайдено
                continue
        
        print(f"⚠️ ПОМИЛКА: Не вдалося знайти жодного зображення в піддиректоріях '{self.output_dir / 'images'}'.")
        print("   Перевірте, чи дані були конвертовані раніше і знаходяться у правильній структурі.")
        return None
