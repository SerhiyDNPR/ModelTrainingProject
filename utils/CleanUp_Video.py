import os
import cv2
import easyocr
import numpy as np
from tqdm import tqdm  # Додаємо tqdm для прогрес-бару

def process_video(video_path, output_path, reader):
    """
    Обробляє один відеофайл: розпізнає текст, пікселізує та розмиває його.

    Args:
        video_path (str): Шлях до вхідного відео.
        output_path (str): Шлях для збереження обробленого відео.
        reader (easyocr.Reader): Екземпляр EasyOCR для розпізнавання тексту.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Помилка: Не вдалося відкрити відео '{video_path}'")
        return False  # Додаємо повернення статусу

    # Отримання властивостей відео
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ініціалізація запису відео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Обробка відео: {os.path.basename(video_path)}")

    stop_flag = False
    # Прогрес-бар для поточного відео
    with tqdm(total=frame_count, desc=f"Кадри {os.path.basename(video_path)}", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Розпізнавання тексту на кадрі
            results = reader.readtext(frame)

            # Обробка кожної знайденої області з текстом
            for (bbox, text, prob) in results:
                # bbox - це координати рамки навколо тексту
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))

                # Вирізання області з текстом
                x, y = top_left
                w, h = bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]

                # Перевірка, чи область не виходить за межі кадру
                if x >= 0 and y >= 0 and x + w <= frame_width and y + h <= frame_height:
                    roi = frame[y:y+h, x:x+w]

                    # 1. Пікселізація
                    # Зменшуємо розмір області, а потім збільшуємо до оригінального
                    pixelated_roi = cv2.resize(roi, (16, 16), interpolation=cv2.INTER_LINEAR)
                    pixelated_roi = cv2.resize(pixelated_roi, (w, h), interpolation=cv2.INTER_NEAREST)

                    # 2. Розмиття (за Гауссом)
                    blurred_roi = cv2.GaussianBlur(pixelated_roi, (25, 25), 0)

                    # Заміна оригінальної області на оброблену
                    frame[y:y+h, x:x+w] = blurred_roi

            # Запис обробленого кадру
            out.write(frame)
            pbar.update(1)

            # --- Відображення кадру в реальному часі ---
            cv2.imshow("Обробка відео (поточний кадр)", frame)

            # --- Додаємо перевірку натискання клавіші Esc ---
            # key = cv2.waitKey(1) & 0xFF
            # if key == 27:  # Esc
            #     print("⏹️ Обробку відео перервано натисканням Esc.")
            #     stop_flag = True
            #     break

    # Звільнення ресурсів
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Відео збережено: {output_path}")
    return stop_flag

def main():
    """
    Головна функція для обробки всіх відео у вказаній директорії.
    """
    # Вкажіть шлях до папки з вашими відео
    directory_path = r"C:\Users\serhi\OneDrive\CD_DSST\Article_syntetic_data\Data_for_tests\Video_interceptors" # <-- ЗАМІНІТЬ ЦЕ ЗНАЧЕННЯ

    # Перевірка існування директорії
    if not os.path.isdir(directory_path):
        print(f"Помилка: Директорія '{directory_path}' не знайдена.")
        return

    # Ініціалізація EasyOCR (може зайняти деякий час при першому запуску)
    # Вказуємо мови для розпізнавання (наприклад, англійська та українська)
    print("Ініціалізація моделі розпізнавання тексту...")
    reader = easyocr.Reader(['en', 'uk'])
    print("Ініціалізація завершена.")

    # Список відеофайлів
    video_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    # Загальний прогрес-бар
    with tqdm(total=len(video_files), desc="Відео", unit="video") as video_pbar:
        for filename in video_files:
            video_path = os.path.join(directory_path, filename)
            # Створення імені для вихідного файлу
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_cleaned{ext}"
            output_path = os.path.join(directory_path, output_filename)

            stop_flag = process_video(video_path, output_path, reader)
            video_pbar.update(1)
            if stop_flag:
                print("⏹️ Обробку всіх відео перервано натисканням Esc.")
                break

if __name__ == "__main__":
    main()