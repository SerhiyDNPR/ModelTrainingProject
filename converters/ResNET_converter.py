import os
import shutil
import json
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict
from converters.converters import BaseDataConverter, remove_readonly

class ResNetDataConverter(BaseDataConverter):
    """Конвертує дані з Unity Perception у формат ImageFolder для ResNet."""

    def prepare_data(self):
        """Головний метод, що запускає процес конвертації для ResNet."""
        print("\n--- Розпочато конвертацію даних у формат ResNet (ImageFolder) ---")
        
        if self.output_dir.exists():
            print(f"🧹 Очищення існуючої директорії: {self.output_dir}")
            shutil.rmtree(self.output_dir, onerror=remove_readonly)

        source_dirs = sorted([p for p in self.source_dir.glob("solo*") if p.is_dir()], key=self._natural_sort_key)
        if not source_dirs:
            print(f"ПОМИЛКА: Не знайдено жодної директорії 'solo*' за шляхом '{self.source_dir}'")
            return

        annotated_dirs = source_dirs[:-1] if len(source_dirs) > 1 else source_dirs
        negative_dir = source_dirs[-1] if len(source_dirs) > 1 else None

        # 1. Збір всіх зображень, їх класів та розміру
        all_image_pairs, imgsz = self._get_image_class_pairs(annotated_dirs, negative_dir) # <-- ЗМІНА 1
        if not all_image_pairs:
            print("ПОМИЛКА: Не знайдено жодного зображення для обробки.")
            return

        # 2. Визначення всіх унікальних класів
        class_names = sorted(list(set(pair['class_name'] for pair in all_image_pairs)))
        print(f"✅ Знайдено {len(class_names)} унікальних класів: {class_names}")

        # 3. Розподіл даних на навчальну, валідаційну та тестову вибірки
        labels = [item['class_name'] for item in all_image_pairs]
        try:
            # Стратифікований розподіл, щоб зберегти пропорції класів
            train_val_files, test_files = train_test_split(all_image_pairs, test_size=0.1, random_state=42, stratify=labels)
            train_labels = [item['class_name'] for item in train_val_files]
            train_files, val_files = train_test_split(train_val_files, test_size=0.111, random_state=42, stratify=train_labels) # 0.111 * 0.9 = ~0.1
        except ValueError:
            # Якщо класів замало для стратифікації, робимо звичайний розподіл
            print("⚠️ Увага: Не вдалося виконати стратифікований розподіл. Використовується звичайний.")
            train_val_files, test_files = train_test_split(all_image_pairs, test_size=0.1, random_state=42)
            train_files, val_files = train_test_split(train_val_files, test_size=0.111, random_state=42)

        splits = {"train": train_files, "val": val_files, "test": test_files}

        # 4. Створення структури папок та копіювання файлів
        self._create_imagefolder_structure(splits, class_names)
        
        # 5. Підрахунок та виведення статистики
        stats_counts = self._calculate_stats(splits)
        print("\n--- ✅ Статистика після конвертації ---")
        print(f"Тренувальна вибірка: {stats_counts['train']['total']} зображень")
        print(f"Валідаційна вибірка: {stats_counts['val']['total']} зображень")
        print(f"Тестова вибірка:    {stats_counts['test']['total']} зображень")
        print("-----------------------------------------")
        print(f"🎉 Загальна кількість зображень у вибірках: {stats_counts['total_unique_images']}")
        print(f"(Примітка: Загальна кількість може бути меншою за суму, якщо зображення належать кільком класам)")

        # 6. Додавання Hard Negatives
        self._add_hard_negatives_resnet()

        print(f"\n🎉 Конвертація даних для ResNet успішно завершена! Результат збережено в: {self.output_dir.resolve()}")
        
        stats = {
            "image_size": imgsz,
            "image_count": stats_counts['total_unique_images'],
            "negative_count": sum(1 for item in all_image_pairs if item['class_name'] == 'background'),
            "class_count": len(class_names)
        }
        return stats

    def get_image_dimensions(self):
        """
        Швидко знаходить розміри зображень. Спочатку шукає в конвертованих даних,
        а якщо їх немає - в вихідних.
        """
        # --- Спроба 1: Пошук в конвертованій директорії (швидший спосіб) ---
        print(f"🔍 Визначення розміру зображень з раніше конвертованих даних у '{self.output_dir}'...")
        
        try:
            from PIL import Image
        except ImportError:
            print("\n⚠️ ПОПЕРЕДЖЕННЯ: Для визначення розміру з файлу потрібна бібліотека Pillow.")
            print("   Будь ласка, встановіть її: pip install Pillow")
            print("   Продовження пошуку у вихідних JSON-файлах...")
        else:
            search_dirs = [
                self.output_dir / "train",
                self.output_dir / "val",
                self.output_dir / "test"
            ]
            for directory in search_dirs:
                if not directory.exists():
                    continue
                try:
                    # Шукаємо перше зображення в будь-якій підпапці класу
                    image_path = next(directory.glob("*/*.[jp][pn]g"))
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
                    return (int(img_w), int(img_h))
            except (StopIteration, json.JSONDecodeError, KeyError):
                continue
        
        print("⚠️ ПОМИЛКА: Не вдалося визначити розмір зображення ані з конвертованих, ані з вихідних файлів.")
        return None

    def _get_image_class_pairs(self, annotated_dirs, negative_dir):
        """Збирає пари (шлях до зображення, назва класу) з усіх джерел."""
        image_class_pairs = []
        imgsz = None # <-- ЗМІНА 3
        
        print("\n🔎 Збір та аналіз файлів з анотаціями...")
        for directory in tqdm(annotated_dirs, desc="Аналіз позитивних прикладів", unit="папка"):
            json_files = [p.parent / "step0.frame_data.json" for p in directory.glob("sequence.*/step0.camera.png") if (p.parent / "step0.frame_data.json").exists()]
            for json_path in json_files:
                img_path = json_path.parent / "step0.camera.png"
                with open(json_path) as f:
                    frame_data = json.load(f)
                
                capture = frame_data.get("capture") or frame_data.get("captures", [{}])[0]
                
                # --> ЗМІНА 4: Отримуємо розмір зображення (лише один раз)
                if imgsz is None and capture.get("dimension"):
                    img_w, img_h = capture["dimension"]
                    imgsz = (int(img_w), int(img_h))
                # <--
                
                annotations_list = frame_data.get("annotations", capture.get("annotations", []))
                
                found_classes = set()
                for annotation in annotations_list:
                    if "BoundingBox2DAnnotation" in annotation.get("@type", ""):
                        for value in annotation.get("values", []):
                            label_name = value.get("label_name") or value.get("labelName")
                            if label_name:
                                found_classes.add(label_name)
                
                if found_classes:
                    for class_name in found_classes:
                        image_class_pairs.append({"img_path": img_path, "class_name": class_name})
        
        print(f"Знайдено {len(image_class_pairs)} позитивних анотацій.")

        if negative_dir:
            print("🔎 Збір файлів з негативними прикладами...")
            all_negative_files = [p for p in negative_dir.glob("sequence.*/step0.camera.png")]
            for img_path in tqdm(all_negative_files, desc="Аналіз негативних прикладів"):
                image_class_pairs.append({"img_path": img_path, "class_name": "background"})
            print(f"Додано {len(all_negative_files)} негативних прикладів до класу 'background'.")
            
        return image_class_pairs, imgsz # <-- ЗМІНА 5

    def _create_imagefolder_structure(self, splits, class_names):
        """Створює структуру папок і копіює зображення."""
        print("\n📦 Формування фінального датасету в форматі ImageFolder...")
        
        for split_name, files in splits.items():
            split_dir = self.output_dir / split_name
            # Створюємо папки для кожного класу
            for class_name in class_names:
                (split_dir / class_name).mkdir(parents=True, exist_ok=True)
            
            # Копіюємо файли
            for item in tqdm(files, desc=f"Копіювання '{split_name}'", unit="file"):
                img_path = item['img_path']
                class_name = item['class_name']
                
                # Створюємо унікальне ім'я файлу, щоб уникнути конфліктів
                parent_folder_name = img_path.parent.parent.name
                sequence_folder_name = img_path.parent.name
                unique_base_name = f"{parent_folder_name}_{sequence_folder_name}.png"
                
                dest_path = split_dir / class_name / unique_base_name
                shutil.copy(img_path, dest_path)

    def _add_hard_negatives_resnet(self):
        """Додає 'складні негативні' приклади до тренувальної вибірки."""
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

        # Цільова папка - train/background
        train_background_dir = self.output_dir / "train" / "background"
        train_background_dir.mkdir(parents=True, exist_ok=True) # Створюємо, якщо її немає

        hn_images = glob(os.path.join(hard_negatives_dir, "*.jpg")) + glob(os.path.join(hard_negatives_dir, "*.png"))
        if not hn_images:
            print("⚠️ У вказаній директорії не знайдено зображень (.jpg або .png).")
            return
            
        print(f"Копіювання {len(hn_images)} Hard Negative файлів до '{train_background_dir}'...")
        for img_path in tqdm(hn_images, desc="Hard Negatives", unit="file"):
            shutil.copy(img_path, train_background_dir / os.path.basename(img_path))
        print(f"✅ Успішно додано {len(hn_images)} файлів до тренувальної вибірки (клас 'background').")

    def _calculate_stats(self, splits):
        """Підраховує кількість файлів у кожній вибірці."""
        stats = defaultdict(lambda: {'total': 0, 'files': set()})
        unique_images = set()

        for split_name, files in splits.items():
            stats[split_name]['total'] = len(files)
            for item in files:
                unique_images.add(item['img_path'])
        
        stats['total_unique_images'] = len(unique_images)
        return stats