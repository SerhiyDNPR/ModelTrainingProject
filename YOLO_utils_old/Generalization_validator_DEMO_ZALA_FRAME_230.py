import cv2
import os
from ultralytics import YOLO

# --- НАЛАШТУВАННЯ ---
MODEL_PATH = r"C:\Users\serhi\OneDrive\CD_DSST\Article_syntetic_data\Trained models\Zala-Supercum-detector-YOLO8n-640-480.pt"
VIDEO_DIR = r"C:\Users\serhi\OneDrive\CD_DSST\Article_syntetic_data\Data_for_tests\Video_interceptors\src"
FILENAME_FILTER = "Зала4_Орлан_Суперкам"
CONF_THRESHOLD = 0.5
EDGE_MARGIN = 150 # Відступ від краю в пікселях для фільтрації об'єктів
# --- КІНЕЦЬ НАЛАШТУВАНЬ ---

# --- Глобальні змінні для керування станом програми ---
is_paused = False
last_results = None
hidden_indices = set()
WINDOW_NAME = "YOLOv8 - Validation"
cap = None
frame = None
model = None

def draw_current_frame():
    """Малює поточний кадр з рамками, написами та інформацією про прогрес."""
    if frame is None:
        return

    annotated_frame = frame.copy()
    img_h, img_w, _ = annotated_frame.shape

    if last_results:
        for i, box in enumerate(last_results[0].boxes):
            if i in hidden_indices:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if (x1 < EDGE_MARGIN or y1 < EDGE_MARGIN or x2 > (img_w - EDGE_MARGIN) or y2 > (img_h - EDGE_MARGIN)):
                continue
            
            conf = box.conf[0]
            cls_id = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            
            class_name = model.names[cls_id]
            label = f"ID:{track_id} {class_name} {conf:.2f}"
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if cap and cap.isOpened():
        current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_text = f"Frame: {current_frame_num} / {total_frames}"
        cv2.putText(annotated_frame, progress_text, (10, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow(WINDOW_NAME, annotated_frame)

def on_mouse_click(event, x, y, flags, param):
    """Обробляє кліки миші для приховування/відображення об'єктів на паузі."""
    global hidden_indices
    
    if frame is not None and y >= frame.shape[0]: return

    if event == cv2.EVENT_LBUTTONDOWN and is_paused and last_results:
        for i, box in enumerate(last_results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x1 < x < x2 and y1 < y < y2:
                if i in hidden_indices:
                    hidden_indices.remove(i)
                else:
                    hidden_indices.add(i)
                print(f"Об'єкт {i} {'приховано' if i in hidden_indices else 'показано'}.")
                draw_current_frame()
                break

def on_trackbar_change(trackbar_value):
    """Обробляє зміну позиції на смузі прокрутки, працюючи лише на паузі."""
    global frame, last_results, hidden_indices
    if is_paused and cap is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, trackbar_value)
        success, new_frame = cap.read()
        if success:
            frame = new_frame
            last_results = model.track(frame, persist=True, conf=CONF_THRESHOLD, verbose=False)
            hidden_indices.clear()
            draw_current_frame()

def load_model(model_path):
    """Завантажує модель YOLO за вказаним шляхом."""
    global model
    try:
        model = YOLO(model_path)
        print(f"✅ Модель '{model_path}' успішно завантажена.")
        return True
    except Exception as e:
        print(f"❌ Помилка завантаження моделі: {e}")
        return False

def find_video_files(video_dir, filename_filter):
    """Знаходить та повертає список шляхів до відеофайлів у вказаній директорії."""
    supported_formats = ['.mp4', '.avi', '.mov']
    try:
        all_files = os.listdir(video_dir)
        video_files = [os.path.join(video_dir, f) for f in all_files if os.path.splitext(f)[1].lower() in supported_formats and filename_filter in f]
        if not video_files:
            print(f"⚠️ У директорії '{video_dir}' не знайдено відеофайлів, що відповідають фільтру '{filename_filter}'.")
        else:
            print(f"🔍 Знайдено {len(video_files)} відео для обробки.")
        return video_files
    except FileNotFoundError:
        print(f"❌ Директорію '{video_dir}' не знайдено.")
        return []

def setup_gui():
    """Створює головне вікно програми та прив'язує обробник подій миші."""
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse_click)

def handle_key_press(key):
    """Обробляє натискання клавіш та повертає прапорці для керування циклом."""
    global is_paused, frame, last_results, hidden_indices
    should_quit = False
    should_next_video = False

    if key in [ord('q'), 27]:
        should_quit = True
    elif key == ord(' '):
        is_paused = not is_paused
        print("⏸️ Пауза" if is_paused else "▶️ Відтворення")
    elif key == ord('n'):
        is_paused = False
        should_next_video = True
    
    if is_paused and key != 255 and key != -1:
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        new_pos = -1

        if key == ord('d'): new_pos = current_pos
        elif key == ord('a'): new_pos = max(0, current_pos - 2)

        if new_pos != -1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            success, frame = cap.read()
            if success:
                last_results = model.track(frame, persist=True, conf=CONF_THRESHOLD, verbose=False)
                hidden_indices.clear()
                if cap and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
                    cv2.setTrackbarPos("Timeline", WINDOW_NAME, new_pos + 1)
                draw_current_frame()
            else:
                should_next_video = True

    return should_quit, should_next_video

def process_video_file(video_path):
    """Виконує головний цикл обробки для одного відеофайлу."""
    global cap, frame, last_results, hidden_indices, is_paused
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Не вдалося відкрити відеофайл: {video_path}")
        cap = None
        return True

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        cv2.createTrackbar("Timeline", WINDOW_NAME, 0, total_frames - 1, on_trackbar_change)
    
    is_paused = False
    
    while cap.isOpened():
        if not is_paused:
            success, frame = cap.read()
            if not success:
                break
            
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if total_frames > 0:
                cv2.setTrackbarPos("Timeline", WINDOW_NAME, current_pos)
            
            last_results = model.track(frame, persist=True, conf=CONF_THRESHOLD, verbose=False)
            hidden_indices.clear()
        
        draw_current_frame()
        key = cv2.waitKey(20) & 0xFF
        
        should_quit, should_next_video = handle_key_press(key)
        
        if should_quit:
            return True
        if should_next_video:
            break
            
    if cap: cap.release()
    cap = None
    return False

if __name__ == "__main__":
    """Головна функція, що керує всім процесом валідації."""
    if not load_model(MODEL_PATH):
        exit

    video_files = find_video_files(VIDEO_DIR, FILENAME_FILTER)
    if not video_files:
        exit

    setup_gui()

    for video_path in video_files:
        print(f"\n▶️ Обробка відео: {video_path}")
        should_quit = process_video_file(video_path)
        if should_quit:
            print("⏹️ Обробку перервано користувачем.")
            break

    cv2.destroyAllWindows()
    print("\n✅ Обробку завершено.")