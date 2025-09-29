import pystac_client
import planetary_computer
import requests
import rasterio
import numpy as np
import os
import random
import glob
from math import cos, radians

def convert_naip_to_jpeg(tiff_path, jpeg_path):
    """Конвертує один 4-канальний NAIP GeoTIFF в кольоровий JPEG."""
    try:
        with rasterio.open(tiff_path) as src:
            if src.count < 3:
                print(f"    ⚠️ Файл '{os.path.basename(tiff_path)}' не є кольоровим. Пропущено.")
                return False
            # ... (решта коду функції без змін)
            red, green, blue = src.read(1), src.read(2), src.read(3)
            profile = src.profile
        def scale(c):
            if c.dtype == np.uint8: return c
            p2, p98 = np.percentile(c[c > 0], (2, 98)); s = np.clip(c, p2, p98)
            return ((s - p2) / (p98 - p2) * 255).astype(np.uint8)
        profile.update(driver='JPEG', count=3, dtype=rasterio.uint8)
        with rasterio.open(jpeg_path, 'w', **profile) as dst:
            dst.write(scale(red), 1); dst.write(scale(green), 2); dst.write(scale(blue), 3)
        print(f"    ✅ Створено кольоровий файл: {os.path.basename(jpeg_path)}")
        return True
    except Exception as e:
        print(f"    ⚠️ Помилка конвертації {os.path.basename(tiff_path)}: {e}")
        return False

def download_file_with_progress(href):
    """Завантажує файл з відображенням прогресу."""
    print(f"⏳ Завантаження файлу: {os.path.basename(href)}")
    response = requests.get(href, stream=True)
    response.raise_for_status()
    # ... (решта коду функції без змін)
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 8192
    temp_data = bytearray()
    for chunk in response.iter_content(chunk_size=chunk_size):
        if chunk:
            temp_data.extend(chunk)
            downloaded = len(temp_data)
            if total_size:
                percent = downloaded * 100 // total_size
                print(f"\r    Прогрес: {percent}% ({downloaded // 1024} KB / {total_size // 1024} KB)", end="")
    print()
    return temp_data

def get_season(month):
    """Визначає сезон за номером місяця."""
    if month in [12, 1, 2]: return "winter"
    if month in [3, 4, 5]: return "spring"
    if month in [6, 7, 8]: return "summer"
    if month in [9, 10, 11]: return "autumn"
    return None

# --- НОВА ФУНКЦІЯ ---
def scan_existing_files(output_dir):
    """Сканує директорію та підраховує кількість існуючих файлів для кожного сезону."""
    counts = {"winter": 0, "spring": 0, "summer": 0, "autumn": 0}
    if not os.path.isdir(output_dir):
        print(f"Директорія '{output_dir}' не існує. Починаємо з нуля.")
        return counts

    # Шукаємо файли, що відповідають шаблону USA_season_number.jpg
    search_pattern = os.path.join(output_dir, "USA_*.jpg")
    for filepath in glob.glob(search_pattern):
        filename = os.path.basename(filepath)
        parts = filename.replace('.jpg', '').split('_')
        if len(parts) == 3 and parts[0] == 'USA':
            season = parts[1]
            if season in counts:
                counts[season] += 1
    return counts

# --- Основна частина скрипту ---
try:
    # 1. Параметри
    USA_BBOX = [-125.0, 24.0, -66.0, 49.0]
    TIME_RANGE = "2020-01-01/2023-12-31"
    IMAGES_PER_SEASON = 10
    # --- ЗМІНЕНО: Шлях до директорії ---
    OUTPUT_DIR = r"C:\Users\serhi\OneDrive\CD_DSST\Article_syntetic_data\Data_for_tests\Map-textures"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- ЗМІНЕНО: Ініціалізуємо лічильники на основі існуючих файлів ---
    season_counts = scan_existing_files(OUTPUT_DIR)
    print("Стан папки перед запуском:")
    for s, c in season_counts.items():
        print(f"  - {s.capitalize()}: знайдено {c} файлів.")

    # 2. Підключення до каталогу
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    print("\n🛰️  Розпочинаємо пошук знімків для доповнення колекції...")

    # 3. Головний цикл пошуку
    while any(count < IMAGES_PER_SEASON for count in season_counts.values()):
        rand_lon = random.uniform(USA_BBOX[0], USA_BBOX[2])
        rand_lat = random.uniform(USA_BBOX[1], USA_BBOX[3])
        search_bbox = [rand_lon, rand_lat, rand_lon + 0.05, rand_lat + 0.05]
        
        status = " | ".join([f"{s.capitalize()}: {c}/{IMAGES_PER_SEASON}" for s, c in season_counts.items()])
        print(f"\n[{status}]")
        print(f"📍 Пошук у новій точці: lon={rand_lon:.2f}, lat={rand_lat:.2f}")

        try:
            search = catalog.search(collections=["naip"], bbox=search_bbox, datetime=TIME_RANGE)
            items = list(search.item_collection())
        except Exception as e:
            print(f"    ⚠️ Помилка пошуку: {e}. Спроба в іншій точці.")
            continue

        if not items:
            print("    - Не знайдено знімків у цій точці.")
            continue
        
        item = items[0]
        item_month = item.datetime.month
        item_season = get_season(item_month)
        
        if item_season and season_counts[item_season] < IMAGES_PER_SEASON:
            print(f"    💡 Знайдено знімок для сезону '{item_season}'! (Поточний стан: {season_counts[item_season]}/{IMAGES_PER_SEASON})")
            
            season_counts[item_season] += 1
            count = season_counts[item_season]
            base_name = f"USA_{item_season}_{count}"
            
            href = item.assets["image"].href
            temp_data = download_file_with_progress(href)
            
            tiff_filename = os.path.join(OUTPUT_DIR, f"{base_name}.tif")
            jpeg_filename = os.path.join(OUTPUT_DIR, f"{base_name}.jpg")

            with open(tiff_filename, "wb") as f:
                f.write(temp_data)

            if convert_naip_to_jpeg(tiff_filename, jpeg_filename):
                os.remove(tiff_filename)
            else:
                print(f"    - Тимчасовий файл {os.path.basename(tiff_filename)} не видалено.")
                season_counts[item_season] -= 1
        else:
            if item_season:
                print(f"    - Знайдено знімок для '{item_season}', але квота вже виконана.")
            else:
                 print(f"    - Не вдалося визначити сезон для знімка.")

    print(f"\n\n🎉 Процес завершено! Всі квоти по 10 файлів на сезон виконано. Файли збережено в '{OUTPUT_DIR}'")

except Exception as e:
    print(f"\n💥 Виникла загальна помилка: {e}")