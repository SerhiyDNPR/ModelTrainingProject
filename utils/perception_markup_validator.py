import json
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from tqdm import tqdm

def verify_unity_perception_data(source_dir: Path):
    """
    Аналізує дані Unity Perception на наявність кадрів без анотацій,
    виводячи статистику для кожної знайденої директорії 'solo*' окремо.

    Args:
        source_dir (Path): Шлях до кореневої директорії з даними.
    """
    print(f"\n🔍 Розпочато верифікацію даних у директорії: {source_dir.resolve()}")

    # Знаходимо всі директорії 'solo*' для окремої обробки
    solo_dirs = sorted([p for p in source_dir.glob("solo*") if p.is_dir()])

    if not solo_dirs:
        print(f"\n❌ ПОМИЛКА: У вказаній директорії не знайдено папок 'solo*'.")
        print("Переконайтеся, що ви обрали правильну папку.")
        return

    # Загальні лічильники для фінального звіту
    grand_total_frames = 0
    grand_total_annotated = 0
    grand_total_unannotated = 0

    # Обробляємо кожну директорію 'solo' окремо
    for solo_dir in solo_dirs:
        print("\n" + "="*50)
        print(f"📁 Аналіз директорії: {solo_dir.name}")
        print("="*50)

        json_files = sorted(list(solo_dir.glob("sequence.*/step0.frame_data.json")))

        if not json_files:
            print("   -> В цій директорії не знайдено кадрів для аналізу.")
            continue

        # Локальні лічильники для поточної директорії
        total_frames = 0
        annotated_frames = 0
        unannotated_frames = 0
        unannotated_files_list = []

        for frame_json_path in tqdm(json_files, desc=f"Перевірка {solo_dir.name}", unit="кадр"):
            total_frames += 1
            has_bbox_annotation = False

            try:
                with open(frame_json_path, 'r', encoding='utf-8') as f:
                    frame_data = json.load(f)

                capture = frame_data.get("capture") or frame_data.get("captures", [{}])[0]
                annotations_list = frame_data.get("annotations", capture.get("annotations", []))

                for annotation in annotations_list:
                    if "BoundingBox2DAnnotation" in annotation.get("@type", "") and annotation.get("values"):
                        has_bbox_annotation = True
                        break
                
                if has_bbox_annotation:
                    annotated_frames += 1
                else:
                    unannotated_frames += 1
                    unannotated_files_list.append(frame_json_path)

            except Exception as e:
                print(f"\n⚠️ Помилка при обробці файлу {frame_json_path}: {e}")
                unannotated_frames += 1
                unannotated_files_list.append(frame_json_path)

        # --- Виведення звіту для поточної директорії ---
        print(f"\n📊 Статистика для '{solo_dir.name}':")
        print(f"   🖼️  Всього кадрів: {total_frames}")
        print(f"   ✅ З розміткою:      {annotated_frames}")
        print(f"   ❗️ БЕЗ розмітки:     {unannotated_frames}")

        if unannotated_files_list:
            print("\n   📋 Список кадрів без розмітки:")
            for file_path in unannotated_files_list:
                # Виводимо шлях відносно поточної 'solo' папки
                relative_path = file_path.relative_to(solo_dir)
                print(f"     - {relative_path}")
        
        # Оновлення загальних лічильників
        grand_total_frames += total_frames
        grand_total_annotated += annotated_frames
        grand_total_unannotated += unannotated_frames

    # --- Виведення фінального загального звіту ---
    print("\n" + "#"*50)
    print("🏆 ЗАГАЛЬНИЙ ПІДСУМОК ПО ВСІХ ДИРЕКТОРІЯХ")
    print("#"*50)
    print(f"📁 Перевірено директорій:   {len(solo_dirs)}")
    print(f"🖼️  Загальна кількість кадрів: {grand_total_frames}")
    print(f"✅ Всього з розміткою:      {grand_total_annotated}")
    print(f"❗️ Всього БЕЗ розмітки:     {grand_total_unannotated}")
    print("#"*50)


if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()

    print("Будь ласка, оберіть кореневу директорію з даними Unity Perception для верифікації...")
    source_directory = filedialog.askdirectory(
        title="Оберіть директорію з даними Unity Perception"
    )

    if not source_directory:
        print("\nДиректорію не обрано. Роботу програми завершено.")
    else:
        source_path = Path(source_directory)
        verify_unity_perception_data(source_path)