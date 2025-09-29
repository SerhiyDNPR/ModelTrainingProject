import os
from PIL import Image
from tqdm import tqdm

def crop_specific_images(input_dir, output_dir):
    """
    Знаходить зображення розміром 640x480 у вхідній папці,
    симетрично обрізає їх до розміру 640x360 і зберігає у вихідну папку.

    Args:
        input_dir (str): Шлях до папки з початковими зображеннями.
        output_dir (str): Шлях до папки для збереження змінених зображень.
    """
    # Створюємо вихідну папку, якщо вона не існує
    os.makedirs(output_dir, exist_ok=True)
    print(f"Вихідна папка '{output_dir}' готова до роботи.")

    # Отримуємо список усіх файлів у вхідній папці
    try:
        all_files = os.listdir(input_dir)
    except FileNotFoundError:
        print(f"Помилка: Вхідну папку '{input_dir}' не знайдено.")
        return

    # Список розширень файлів, які вважаються зображеннями
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    
    # Фільтруємо лише файли зображень
    image_files = [f for f in all_files if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print("У вхідній папці не знайдено файлів зображень.")
        return

    print(f"Знайдено {len(image_files)} зображень. Починаю обробку...")
    processed_count = 0

    # Координати для обрізки (left, upper, right, lower)
    # Щоб з 480 зробити 360, треба відрізати 120 пікселів (60 зверху, 60 знизу)
    # Верхня межа: 60
    # Нижня межа: 480 - 60 = 420
    crop_box = (0, 60, 640, 420)

    # Обробляємо кожен файл зображення з індикатором прогресу
    for filename in tqdm(image_files, desc="Обрізка зображень"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            with Image.open(input_path) as img:
                # Перевіряємо, чи розмір зображення 640x480
                if img.size == (640, 480):
                    # Обрізаємо зображення
                    cropped_img = img.crop(crop_box)
                    # Зберігаємо у вихідну папку
                    cropped_img.save(output_path)
                    processed_count += 1
        except Exception as e:
            print(f"\nНе вдалося обробити файл '{filename}'. Помилка: {e}")
            
    print("\n-----------------------------------------")
    print("✅ Обробку завершено.")
    print(f"Знайдено та обрізано файлів з розміром 640x480: {processed_count}")
    print(f"Збережено у папку: {output_dir}")
    print("-----------------------------------------")


if __name__ == '__main__':
    # Запитуємо у користувача шляхи до папок
    input_folder = r"C:\Users\serhi\OneDrive\CD_DSST\Article_syntetic_data\Data_for_tests\Hard_Negatives\YOLO\248+117 = 336 (some are duplicates)-3x4"
    output_folder = r"C:\Users\serhi\OneDrive\CD_DSST\Article_syntetic_data\Data_for_tests\Hard_Negatives\YOLO\248+117 = 336 (some are duplicates)-9x16"

    crop_specific_images(input_folder, output_folder)