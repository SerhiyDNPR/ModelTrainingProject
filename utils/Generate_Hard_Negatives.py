import cv2
import os
from ultralytics import YOLO

# --- 1. НАЛАШТУВАННЯ ---
# Шлях до вашої натренованої моделі YOLO
MODEL_PATH = r"C:\Users\serhi\OneDrive\CD_DSST\Article_syntetic_data\Trained models\Zala-Supercum-detector-YOLO8n-640-480-2000-samples-40Epochs-with-117-Hard-Negatives_and_DeFocus.pt"
# Каталог з відео для аналізу
VIDEO_DIR = r"C:\Users\serhi\OneDrive\CD_DSST\Article_syntetic_data\Data_for_tests\Video_interceptors\src"
# Вихідний каталог для збереження негативних прикладів
OUTPUT_DIR = r"C:\Users\serhi\OneDrive\CD_DSST\Article_syntetic_data\Data_for_tests\Hard_Negatives"
# Фільтр для імен відеофайлів (наприклад, "Drone" або "*" для всіх)
FILENAME_FILTER = "*"
CROP_SIZE = (640, 480)
FRAME_STEP = 40

crop_center = None
WINDOW_NAME = "Hard Negative Mining Tool"
# Коди клавіш
KEY_ENTER = 13
KEY_SPACE = 32
KEY_ESC = 27
KEY_Q = ord('q')

def mouse_callback(event, x, y, flags, param):
    """Обробляє кліки миші, зберігаючи центр для майбутньої обрізки."""
    global crop_center
    if event == cv2.EVENT_LBUTTONDOWN:
        crop_center = (x, y)
        print(f"Обрано новий центр для вирізання: {crop_center}")

def save_negative_sample(frame, rect, output_dir, counter):
    """Зберігає вирізану область та створює порожній файл розмітки."""
    x1, y1, x2, y2 = rect
    cropped_image = frame[y1:y2, x1:x2]

    base_filename = f"Negative_{counter}"
    image_path = os.path.join(output_dir, f"{base_filename}.jpg")
    label_path = os.path.join(output_dir, f"{base_filename}.txt")

    cv2.imwrite(image_path, cropped_image)
    with open(label_path, 'w') as f:
        pass

    print(f"✅ Збережено: {image_path} та {label_path}")
    return counter + 1

def find_video_files(video_dir, filename_filter):
    """Знаходить відеофайли у вказаному каталозі, що відповідають фільтру."""
    supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    print(f"🔍 Пошук відео у '{video_dir}'...")
    if not os.path.isdir(video_dir):
        print(f"❌ Каталог не знайдено: {video_dir}")
        return []
        
    for f in os.listdir(video_dir):
        name_matches = (filename_filter == '*') or (filename_filter in f)
        
        if os.path.splitext(f)[1].lower() in supported_formats and name_matches:
            video_files.append(os.path.join(video_dir, f))
            
    print(f"Знайдено {len(video_files)} відеофайлів.")
    return video_files

def get_start_counter(output_dir):
    """Визначає початковий номер для іменування файлів, щоб уникнути перезапису."""
    os.makedirs(output_dir, exist_ok=True)
    existing_files = os.listdir(output_dir)
    max_num = 0
    if existing_files:
        for f in existing_files:
            if f.startswith("Negative_") and f.endswith(".jpg"):
                try:
                    num = int(f.replace("Negative_", "").replace(".jpg", ""))
                    if num > max_num:
                        max_num = num
                except ValueError:
                    continue
    start_counter = max_num + 1
    print(f"Продовжуємо нумерацію з {start_counter}")
    return start_counter

def initialize_app():
    """Завантажує модель та налаштовує вікно OpenCV."""
    print("Завантаження моделі YOLO...")
    try:
        model = YOLO(MODEL_PATH)
        print("✅ Модель успішно завантажена.")
    except Exception as e:
        print(f"❌ Помилка завантаження моделі: {e}")
        return None
    
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    return model

def handle_frame_interaction(original_frame, annotated_frame, counter):
    """Обробляє взаємодію користувача з одним кадром (кліки, натискання клавіш)."""
    global crop_center
    
    while True:
        display_frame = annotated_frame.copy()
        rect_to_save = None

        if crop_center:
            h, w, _ = display_frame.shape
            crop_w, crop_h = CROP_SIZE
            x1 = max(0, crop_center[0] - crop_w // 2)
            y1 = max(0, crop_center[1] - crop_h // 2)
            x2 = min(w, x1 + crop_w)
            y2 = min(h, y1 + crop_h)
            if x2 == w: x1 = w - crop_w
            if y2 == h: y1 = h - crop_h
            rect_to_save = (x1, y1, x2, y2)
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(display_frame, "Selected Area", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == KEY_ENTER:
            if rect_to_save:
                counter = save_negative_sample(original_frame, rect_to_save, OUTPUT_DIR, counter)
                crop_center = None
                return counter, "next_frame" 
            else:
                print("⚠️ Спочатку клікніть мишею, щоб обрати область!")
        elif key == KEY_SPACE:
            crop_center = None
            return counter, "next_frame"
        elif key in [KEY_Q, KEY_ESC]:
            return counter, "quit"

def process_video(video_path, model, counter):
    """Обробляє один відеофайл кадр за кадром."""
    global crop_center
    print(f"\n▶️ Обробка відео: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Не вдалося відкрити відео: {video_path}")
        return counter, False

    frame_idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success:
            print("🏁 Завершено обробку відео.")
            break

        print(f"--- Обробка кадру {frame_idx} ---")
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        counter, status = handle_frame_interaction(frame, annotated_frame, counter)
        
        if status == "next_frame":
            frame_idx += FRAME_STEP
        elif status == "quit":
            cap.release()
            return counter, True

    cap.release()
    return counter, False

def main():
    """Головна точка входу. Керує загальним процесом."""
    model = initialize_app()
    if not model:
        return

    video_files = find_video_files(VIDEO_DIR, FILENAME_FILTER)
    if not video_files:
        return
        
    negative_counter = get_start_counter(OUTPUT_DIR)

    should_quit = False
    for video_path in video_files:
        negative_counter, should_quit = process_video(video_path, model, negative_counter)
        if should_quit:
            break
    
    if should_quit:
        print("⏹️ Вихід з програми.")
    else:
        print("\n🎉 Всі відеофайли оброблено.")
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()