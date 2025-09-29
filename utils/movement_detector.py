import cv2
import numpy as np
import supervision as sv
from tkinter import Tk
from tkinter.filedialog import askopenfilename

is_paused = False  # Відповідає за паузу
cap = None         # Зробимо об'єкт відео глобальним для доступу з колбеку

def setup_tracker():
    """Налаштовує та повертає трекер ByteTrack з налаштуваннями за замовчуванням."""
    return sv.ByteTrack()

def on_trackbar_change(frame_number):
    """Ця функція викликається, коли користувач рухає трекбар."""
    global is_paused
    if cap is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        is_paused = False

def main():
    """Основна функція для детектування руху та відстеження об'єктів у відео."""
    global is_paused, cap

    Tk().withdraw()
    video_path = askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )

    if not video_path:
        print("No file selected. Exiting.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    window_name = "Motion Detector and Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        cv2.createTrackbar("Frame", window_name, 0, total_frames - 1, on_trackbar_change)

    back_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=75, detectShadows=False)
    byte_tracker = setup_tracker()
    
    bounding_box_annotator = sv.RoundBoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, color_lookup=sv.ColorLookup.TRACK)

    # Ініціалізуємо змінні для зберігання останнього кадру
    annotated_frame = None
    fg_mask = None

    while cap.isOpened():
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                is_paused = True
                while is_paused:
                    key = cv2.waitKey(100) & 0xFF
                    if key == 27:
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    elif key == ord(' '):
                         is_paused = not is_paused
                continue

            current_fg_mask = back_sub.apply(frame)
            current_fg_mask = cv2.erode(current_fg_mask, None, iterations=3)
            current_fg_mask = cv2.dilate(current_fg_mask, None, iterations=3)
            fg_mask = cv2.threshold(current_fg_mask, 127, 255, cv2.THRESH_BINARY)[1] # Зберігаємо маску для показу

            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detections_list = []
            for contour in contours:
                if cv2.contourArea(contour) < 900:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                detections_list.append([x, y, x + w, y + h])

            detections_np = np.array(detections_list)
            
            detections = sv.Detections.empty()
            if detections_np.size > 0:
                confidence_scores = np.ones(len(detections_np))
                detections = sv.Detections(
                    xyxy=detections_np,
                    confidence=confidence_scores
                )
            
            tracked_objects = byte_tracker.update_with_detections(detections)

            labels = []
            if tracked_objects.tracker_id is not None:
                 labels = [f"ID: {tracker_id}" for tracker_id in tracked_objects.tracker_id]
            
            # Анотуємо та зберігаємо кадр для відображення
            annotated_frame = frame.copy()
            annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=tracked_objects)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_objects, labels=labels)

            # Оновлення позиції трекбару
            if total_frames > 0:
                current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cv2.setTrackbarPos("Frame", window_name, current_frame_pos)

        if annotated_frame is not None:
            cv2.imshow(window_name, annotated_frame)
        if fg_mask is not None:
            cv2.imshow("Motion Mask", fg_mask)
        
        key = cv2.waitKey(30) & 0xFF
        if key == 27: # ESC для виходу
            break
        elif key == ord(' '): # Пробіл для паузи/відтворення
            is_paused = not is_paused

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()