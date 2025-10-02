import os
import shutil
import json
from pathlib import Path
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
import collections

def remove_readonly(func, path, _):
    """Допоміжна функція для видалення файлів лише для читання."""
    os.chmod(path, 0o777)
    func(path)

class COCODataConverter:
    """Конвертує дані з Unity Perception у формат COCO, сумісний з DETR."""
    
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Вихідна директорія не знайдена: {self.source_dir}")

    def _natural_sort_key(self, s):
        """Допоміжна функція для природного сортування."""
        import re
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]
    
    def get_image_dimensions(self):
        """
        Швидко знаходить розміри зображень. Спочатку шукає в конвертованих даних,
        а якщо їх немає - в вихідних.
        """
        print(f"🔍 Визначення розміру зображень з раніше конвертованих даних у '{self.output_dir}'...")
        
        try:
            from PIL import Image
        except ImportError:
            print("\n⚠️ ПОПЕРЕДЖЕННЯ: Для визначення розміру з файлу потрібна бібліотека Pillow.")
            print("   Будь ласка, встановіть її: pip install Pillow")
            print("   Продовження пошуку у вихідних JSON-файлах...")
        else:
            search_dirs = [self.output_dir / "train", self.output_dir / "val"]
            for directory in search_dirs:
                if not directory.exists():
                    continue
                try:
                    # Шукаємо перше-ліпше зображення
                    image_path = next(directory.glob("*.*"))
                    # Перевіряємо, чи це справді зображення
                    if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        with Image.open(image_path) as img:
                            width, height = img.size
                            print(f"✅ Розмір зображення визначено: {width}x{height} (з файлу {image_path.name})")
                            return (width, height)
                except (StopIteration, OSError):
                    continue
        
        print(f"⚠️  Не вдалося знайти зображення в '{self.output_dir}'. Спроба пошуку у вихідних даних...")

        # --- Спроба 2: Пошук у вихідних даних (з JSON) ---
        source_dirs_list = [p for p in self.source_dir.glob("solo*") if p.is_dir()]
        if not source_dirs_list:
            print(f"ПОМИЛКА: Не знайдено папок 'solo*' в {self.source_dir}")
            return None

        for directory in source_dirs_list:
            try:
                frame_data_path = next(directory.glob("sequence.*/step0.frame_data.json"))
                with open(frame_data_path) as f:
                    frame_data = json.load(f)
                
                capture = frame_data.get("capture") or frame_data.get("captures", [{}])[0]
                img_w, img_h = capture.get("dimension", [None, None])

                if img_w and img_h:
                    print(f"✅ Розмір зображення визначено з вихідних даних: {img_w}x{img_h}")
                    return (img_w, img_h)
            except (StopIteration, json.JSONDecodeError, KeyError):
                continue
        
        print("⚠️ ПОМИЛКА: Не вдалося визначити розмір зображення ані з конвертованих, ані з вихідних файлів.")
        return None


    def prepare_data(self):
        """Головний метод для запуску процесу конвертації."""
        print("\n--- Розпочато конвертацію даних у формат COCO (для DETR) ---")

        if self.output_dir.exists():
            print(f"🧹 Очищення існуючої директорії: {self.output_dir}")
            shutil.rmtree(self.output_dir, onerror=remove_readonly)
        
        # Створення нової структури папок
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        self.annotations_dir = self.output_dir / "annotations"
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

        source_dirs = sorted([p for p in self.source_dir.glob("solo*") if p.is_dir()], key=self._natural_sort_key)
        if not source_dirs:
            print(f"ПОМИЛКА: Не знайдено жодної директорії 'solo*' за шляхом '{self.source_dir}'")
            return None

        annotated_dirs = source_dirs[:-1] if len(source_dirs) > 1 else source_dirs
        negative_dir = source_dirs[-1] if len(source_dirs) > 1 else None

        # 1. Сканування класів
        print("🔍 Сканування класів у вихідних даних...")
        class_names = self._discover_classes(annotated_dirs)
        categories = [{"id": i, "name": name, "supercategory": "object"} for i, name in enumerate(class_names, 1)]
        class_map = {name: i for i, name in enumerate(class_names, 1)}
        print(f"✅ Знайдено {len(class_names)} унікальних класів: {class_names}")

        # 2. Збір усіх даних
        all_examples = []
        self._gather_annotated_data(annotated_dirs, class_map, all_examples)
        negative_files_count = self._gather_negative_data(negative_dir, all_examples)

        # 3. Розділення даних на тренувальну та валідаційну вибірки
        if not all_examples:
            print("⚠️ Не знайдено жодних прикладів для обробки. Завершення роботи.")
            return None
        
        # Визначаємо розмір зображення з першого прикладу
        image_size = (all_examples[0]['width'], all_examples[0]['height'])
            
        train_data, val_data = train_test_split(all_examples, test_size=0.2, random_state=42)
        print(f"\n📊 Розподіл даних: {len(train_data)} для тренування, {len(val_data)} для валідації.")

        # Додавання 'складних негативів' до тренувальної вибірки
        self._add_hard_negatives(train_data)

        # 4. Створення COCO JSON файлів
        print("\n📦 Створення файлів анотацій у форматі COCO...")
        self._create_coco_json(train_data, self.train_dir, self.annotations_dir / "instances_train.json", categories)
        self._create_coco_json(val_data, self.val_dir, self.annotations_dir / "instances_val.json", categories)
        
        print("\n🎉 Конвертація даних для DETR успішно завершена!")

        stats = {
            "image_size": image_size,
            "image_count": len(all_examples),
            "negative_count": negative_files_count,
            "class_count": len(class_names)
        }
        return stats


    def _discover_classes(self, annotated_dirs):
        """Сканує всі JSON-файли для виявлення унікальних назв класів та розрахунку статистики."""
        print("🔍 Сканування класів та збір статистики у вихідних даних Perception...")
        class_names = set()
        
        stats_dir = self.output_dir / "statistics"
        stats_dir.mkdir(parents=True, exist_ok=True)

        for directory in tqdm(annotated_dirs, desc="Пошук класів", unit="папка"):
            dir_stats = collections.defaultdict(list)
            
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

    def _gather_annotated_data(self, annotated_dirs, class_map, all_examples):
        """Збирає інформацію з анотованих директорій."""
        print("\n🔎 Аналіз позитивних прикладів...")
        for directory in tqdm(annotated_dirs, desc="Обробка папок", unit="папка"):
            all_files = sorted(
                [(p.parent / "step0.frame_data.json", p.parent / "step0.camera.png")
                for p in directory.glob("sequence.*/step0.camera.png") if (p.parent / "step0.frame_data.json").exists()],
                key=lambda x: int(x[1].parent.name.split('.')[-1])
            )
            for json_path, img_path in all_files:
                with open(json_path) as f:
                    frame_data = json.load(f)
                
                capture = frame_data.get("capture") or frame_data.get("captures", [{}])[0]
                img_w, img_h = capture.get("dimension", [0, 0])
                
                annotations = []
                annotations_list = frame_data.get("annotations", capture.get("annotations", []))
                for annotation in annotations_list:
                    if "BoundingBox2DAnnotation" in annotation.get("@type", ""):
                        for value in annotation.get("values", []):
                            class_name = value.get("label_name") or value.get("labelName")
                            class_id = class_map.get(class_name)
                            if class_id is None: continue
                            
                            px_x, px_y = value["origin"]
                            px_w, px_h = value["dimension"]
                            
                            annotations.append({
                                "category_id": class_id,
                                "bbox": [float(px_x), float(px_y), float(px_w), float(px_h)]
                            })
                
                if annotations:
                    all_examples.append({
                        "img_path": img_path,
                        "width": img_w,
                        "height": img_h,
                        "annotations": annotations
                    })
        print(f"Знайдено {len(all_examples)} позитивних прикладів з анотаціями.")

    def _gather_negative_data(self, negative_dir, all_examples):
        """Збирає інформацію про негативні приклади (без анотацій)."""
        if not negative_dir:
            return 0
        
        print("\n🔎 Аналіз негативних прикладів...")
        negative_files = [p for p in negative_dir.glob("sequence.*/step0.camera.png")]
        for img_path in tqdm(negative_files, desc="Обробка негативів", unit="file"):
            # Для негативних прикладів нам потрібні розміри зображення.
            # Якщо вони не вказані, їх потрібно буде отримати, наприклад, з Pillow.
            # Тут ми припускаємо, що всі зображення однакового розміру.
            # Якщо ні, потрібна додаткова логіка.
            # У цьому прикладі ми беремо розміри з останнього позитивного прикладу,
            # але краще було б відкрити зображення і отримати реальні розміри.
            last_dims = (all_examples[-1]['width'], all_examples[-1]['height']) if all_examples else (1920, 1080)

            all_examples.append({
                "img_path": img_path,
                "width": last_dims[0],
                "height": last_dims[1],
                "annotations": [] # Порожній список анотацій
            })
        print(f"Додано {len(negative_files)} негативних прикладів.")
        return len(negative_files)

    def _add_hard_negatives(self, train_data):
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

        hn_images = glob(os.path.join(hard_negatives_dir, "*.jpg")) + glob(os.path.join(hard_negatives_dir, "*.png"))
        if not hn_images:
            print("⚠️ У вказаній директорії не знайдено зображень (.jpg або .png).")
            return
            
        print(f"Додавання {len(hn_images)} Hard Negative файлів...")
        # Для отримання розмірів зображень потрібен Pillow
        try:
            from PIL import Image
        except ImportError:
            print("\nПОПЕРЕДЖЕННЯ: Для обробки Hard Negatives потрібна бібліотека Pillow.")
            print("Будь ласка, встановіть її: pip install Pillow")
            return

        for img_path in tqdm(hn_images, desc="Hard Negatives", unit="file"):
            img_path = Path(img_path)
            with Image.open(img_path) as img:
                width, height = img.size
            
            train_data.append({
                "img_path": img_path,
                "width": width,
                "height": height,
                "annotations": []
            })
        print(f"✅ Успішно додано {len(hn_images)} файлів до тренувальної вибірки.")

    def _create_coco_json(self, data, dest_img_dir, json_path, categories):
        """Створює єдиний JSON-файл у форматі COCO для заданого набору даних."""
        
        coco_output = {
            "info": {"description": "Dataset created from Unity Perception"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": categories
        }

        image_id_counter = 1
        annotation_id_counter = 1

        for item in tqdm(data, desc=f"Створення {json_path.name}", unit="file"):
            # Унікальне ім'я файлу, щоб уникнути конфліктів
            img_path = Path(item['img_path'])
            parent_folder_name = img_path.parent.parent.name
            sequence_folder_name = img_path.parent.name
            unique_base_name = f"{parent_folder_name}_{sequence_folder_name}_{img_path.name}"
            
            # Копіювання зображення
            shutil.copy(img_path, dest_img_dir / unique_base_name)
            
            # Додавання запису про зображення
            image_info = {
                "id": image_id_counter,
                "file_name": unique_base_name,
                "width": item['width'],
                "height": item['height']
            }
            coco_output["images"].append(image_info)
            
            # Додавання записів про анотації
            for ann in item['annotations']:
                bbox = ann['bbox']
                area = bbox[2] * bbox[3] # width * height
                
                annotation_info = {
                    "id": annotation_id_counter,
                    "image_id": image_id_counter,
                    "category_id": ann['category_id'],
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": [] # Залишаємо порожнім для детекції
                }
                coco_output["annotations"].append(annotation_info)
                annotation_id_counter += 1
            
            image_id_counter += 1
            
        # Збереження фінального JSON файлу
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_output, f, ensure_ascii=False, indent=4)
        
        print(f"✅ Файл '{json_path.name}' створено. Зображень: {len(coco_output['images'])}, анотацій: {len(coco_output['annotations'])}.")


if __name__ == '__main__':
    # --- ЯК ВИКОРИСТОВУВАТИ ---
    # 1. Вкажіть шлях до папки, що містить ваші дані з Unity (де лежать папки solo_0, solo_1 і т.д.)
    # 2. Вкажіть шлях до папки, куди буде збережено сконвертований датасет.

    # Використовуємо діалогове вікно для вибору папок
    root = tk.Tk()
    root.withdraw()

    print("Будь ласка, оберіть кореневу директорію з даними Unity Perception...")
    source_directory = filedialog.askdirectory(title="Оберіть директорію з даними Unity Perception")
    if not source_directory:
        print("Директорію не обрано. Вихід.")
    else:
        print(f"Обрано вихідну директорію: {source_directory}")

        print("\nБудь ласка, оберіть директорію для збереження COCO датасету...")
        output_directory = filedialog.askdirectory(title="Оберіть директорію для збереження результату")
        if not output_directory:
            print("Директорію не обрано. Вихід.")
        else:
            print(f"Обрано директорію для збереження: {output_directory}")

            source_path = Path(source_directory)
            output_path = Path(output_directory)

            converter = COCODataConverter(source_dir=source_path, output_dir=output_path)
            converter.prepare_data()