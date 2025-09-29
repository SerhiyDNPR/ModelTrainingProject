# video_processor.py

import cv2
import os
import numpy as np
import supervision as sv
from collections import namedtuple
import json

Prediction = namedtuple('Prediction', ['box', 'score', 'class_name', 'track_id'])

class VideoProcessor:
    """
    Керує логікою обробки відео, GUI та логування.
    Фільтрація: одинарний клік - тимчасовий фільтр, подвійний - постійний.
    """
    def __init__(self, loaded_models: list, window_name: str, conf_threshold: float, initial_filters=None):
        self.loaded_models = loaded_models
        self.window_name = window_name
        self.conf_threshold = conf_threshold
        self.trackers = []
        self.log_records = []
        
        self.filters = []
        self.temp_filters = []
        self.any_model_uses_tracker = any(m.get('use_tracker', False) for m in self.loaded_models)
        
        self.ignore_regions = [
            (0, 0, 180, 180),
            (1130, 0, 1300, 180),
            (515, 410, 795, 735)
        ]
        
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
        ]
        
        self.cap = None
        self.frame = None
        self.is_paused = False
        self.last_predictions_all_models = []
        
        self.is_drawing = False
        self.drawing_start_point = None
        self.drawing_end_point = None
        
        self.filters_filepath = "filters.json"
        self.dynamic_ignore_regions = []
        if initial_filters is not None:
            self.dynamic_ignore_regions = initial_filters
            print(f"✅ Завантажено {len(self.dynamic_ignore_regions)} зон фільтрації з файлу.")
        
        self.KEY_LEFT_ARROW = 2424832
        self.KEY_RIGHT_ARROW = 2555904

    def _save_filters_to_json(self):
        """Зберігає поточний список динамічних зон у файл JSON."""
        try:
            with open(self.filters_filepath, 'w', encoding='utf-8') as f:
                json.dump(self.dynamic_ignore_regions, f, indent=4)
            print(f"💾 Фільтри збережено у файл {self.filters_filepath}")
        except Exception as e:
            print(f"❌ Помилка збереження фільтрів: {e}")

    def _initialize_trackers(self):
        """Скидає та ініціалізує трекери для кожної моделі."""
        self.trackers = []
        for _ in self.loaded_models:
            tracker = sv.ByteTrack(frame_rate=30, lost_track_buffer=30)
            self.trackers.append(tracker)
        print(f"🔄️ Ініціалізовано {len(self.trackers)} трекерів ByteTrack (supervision).")

    def _is_box_in_ignore_region(self, box):
        bx1, by1, bx2, by2 = map(int, box)
        all_regions = self.ignore_regions + self.dynamic_ignore_regions
        for r in all_regions:
            rx1, ry1 = min(r[0], r[2]), min(r[1], r[3])
            rx2, ry2 = max(r[0], r[2]), max(r[1], r[3])
            if (bx1 >= rx1 and by1 >= ry1 and bx2 <= rx2 and by2 <= ry2):
                return True
        return False
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Обробляє кліки та малювання мишею для створення зон фільтрації."""
        if not self.is_paused:
            return

        if event == cv2.EVENT_LBUTTONDBLCLK:
            print("\n🔄 Подвійний клік. Перезапуск обробки з поточними фільтрами...")
            self._restart_processing()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.drawing_start_point = (x, y)
            self.drawing_end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing:
                self.drawing_end_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_drawing:
                self.is_drawing = False
                final_rect = (*self.drawing_start_point, *self.drawing_end_point)
                if final_rect not in self.dynamic_ignore_regions:
                    self.dynamic_ignore_regions.append(final_rect)
                    print(f"✅ Додано нову унікальну зону фільтрації: {final_rect}")
                    self._save_filters_to_json() # Зберігаємо у файл
                else:
                    print("ℹ️ Ця зона вже існує у фільтрах.")

                current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_pos - 1))
                self._process_single_frame()

    def _restart_processing(self):
        """Перезапускає обробку відео з самого початку."""
        self.log_records.clear()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        if self.any_model_uses_tracker:
            self._initialize_trackers()
            
        self.is_paused = False
        print("▶️ Відтворення відновлено.")

    def _is_object_filtered(self, pred):
        """Перевіряє, чи об'єкт відповідає будь-якому з активних фільтрів."""
        px1, py1, px2, py2 = map(int, pred.box)
        p_width = px2 - px1
        p_height = py2 - py1

        all_filters = self.filters + self.temp_filters
        for f in all_filters:
            tol = f['tolerance']
            if (abs(px1 - f['x1']) <= tol and
                abs(py1 - f['y1']) <= tol and
                abs(p_width - f['width']) <= tol and
                abs(p_height - f['height']) <= tol):
                return True
        return False

    def find_video_files(self, video_dir, filename_filter):
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
    
    def _log_frame_result(self, frame_number):
        log_entry = {'FrameNumber': frame_number}
        for i, (model_info, preds) in enumerate(zip(self.loaded_models, self.last_predictions_all_models)):
            model_id = i + 1
            log_entry[f'ModelFile_{model_id}'] = model_info['filename']
            log_entry[f'ModelType_{model_id}'] = model_info['name']
            
            if preds:
                best_pred = max(preds, key=lambda p: p.score)
                log_entry[f'ObjectName_{model_id}'] = best_pred.class_name
                log_entry[f'Probability_{model_id}'] = f"{best_pred.score:.4f}"
                x1, y1, x2, y2 = map(int, best_pred.box)
                width, height = x2 - x1, y2 - y1
                log_entry[f'Width_{model_id}'], log_entry[f'Height_{model_id}'] = width, height
                log_entry[f'TrackID_{model_id}'] = best_pred.track_id if best_pred.track_id is not None else ''
            else:
                log_entry[f'ObjectName_{model_id}'] = "No_Object"
                log_entry[f'Probability_{model_id}'] = 0.0
                log_entry[f'Width_{model_id}'], log_entry[f'Height_{model_id}'] = 0, 0
                log_entry[f'TrackID_{model_id}'] = ''
        
        self.log_records.append(log_entry)

    def _draw_legend(self, frame):
        start_y = 30
        for i, model_info in enumerate(self.loaded_models):
            color = self.colors[i % len(self.colors)]
            tracker_marker = " (T)" if model_info.get('use_tracker', False) else ""
            text = f"{model_info['filename']}{tracker_marker}"
            cv2.rectangle(frame, (10, start_y - 15), (25, start_y), color, -1)
            cv2.putText(frame, text, (35, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            start_y += 30
        return frame

    def _draw_frame(self):
        if self.frame is None: return
        
        annotated_frame = self.frame.copy()
        img_h, img_w, _ = annotated_frame.shape

        if self.is_drawing and self.drawing_start_point and self.drawing_end_point:
            cv2.rectangle(annotated_frame, self.drawing_start_point, self.drawing_end_point, (0, 255, 255), 2)

        for i, model_preds in enumerate(self.last_predictions_all_models):
            color = self.colors[i % len(self.colors)]
            for pred in model_preds:
                x1, y1, x2, y2 = map(int, pred.box)
                label = f"{pred.class_name} {pred.score:.2f}"
                if pred.track_id is not None:
                    label = f"ID:{pred.track_id} " + label
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        annotated_frame = self._draw_legend(annotated_frame)
        current_frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_text = f"Frame: {current_frame_num} / {total_frames}"
        cv2.putText(annotated_frame, progress_text, (10, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(self.window_name, annotated_frame)
    
    def _handle_key_press(self, key):
        should_quit, should_next_video = False, False
        
        if key in [ord('q'), 27]: 
            should_quit = True
        elif key == ord(' '):
            self.is_paused = not self.is_paused
            print("⏸️ Пауза" if self.is_paused else "▶️ Відтворення")
        elif key == ord('n'):
            self.is_paused = False
            self.temp_filters.clear()
            self.dynamic_ignore_regions.clear()
            should_next_video = True
        
        if self.is_paused and key != -1:
            if key == ord('r'):
                self.temp_filters.clear()
                self.dynamic_ignore_regions.clear()
                print("🗑️ Очищено тимчасові фільтри та намальовані зони.")
                current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_pos - 1))
                self._process_single_frame()
                return should_quit, should_next_video

            current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            new_pos = -1
            
            if key in [self.KEY_RIGHT_ARROW, self.KEY_LEFT_ARROW]:
                if key == self.KEY_RIGHT_ARROW: 
                    new_pos = current_pos
                elif key == self.KEY_LEFT_ARROW: 
                    new_pos = max(0, current_pos - 2)
            
            if new_pos != -1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                if not self._process_single_frame():
                    should_next_video = True
                
        return should_quit, should_next_video

    def _process_single_frame(self):
        success, self.frame = self.cap.read()
        if not success:
            return False

        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.last_predictions_all_models = []

        for i, model_info in enumerate(self.loaded_models):
            preds = model_info['wrapper'].predict(self.frame, self.conf_threshold)
            
            preds_after_ignore = [p for p in preds if not self._is_box_in_ignore_region(p.box)]
            preds_after_dynamic_filter = [p for p in preds_after_ignore if not self._is_object_filtered(p)]

            if model_info.get('use_tracker', False) and self.trackers:
                if not preds_after_dynamic_filter:
                    detections = sv.Detections.empty()
                else:
                    boxes = np.array([p.box for p in preds_after_dynamic_filter])
                    scores = np.array([p.score for p in preds_after_dynamic_filter])
                    class_names_data = model_info['wrapper'].class_names
                    if isinstance(class_names_data, dict):
                        class_names_list = list(class_names_data.values())
                    else:
                        class_names_list = class_names_data

                    class_ids = np.array([class_names_list.index(p.class_name) for p in preds_after_dynamic_filter])
                    detections = sv.Detections(xyxy=boxes, confidence=scores, class_id=class_ids)

                tracked_detections = self.trackers[i].update_with_detections(detections)
                preds_for_log = []
                for box, score, class_id, tracker_id in zip(tracked_detections.xyxy, tracked_detections.confidence, tracked_detections.class_id, tracked_detections.tracker_id):
                    preds_for_log.append(Prediction(box=box, score=score, class_name=class_names_list[class_id], track_id=tracker_id))
                self.last_predictions_all_models.append(preds_for_log)
            else:
                preds_with_none_track_id = [Prediction(box=p.box, score=p.score, class_name=p.class_name, track_id=None) for p in preds_after_dynamic_filter]
                self.last_predictions_all_models.append(preds_with_none_track_id)

        self._log_frame_result(current_pos)
        
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            normalized_pos = int((current_pos / total_frames) * 1000)
            cv2.setTrackbarPos("Timeline", self.window_name, normalized_pos)

        self._draw_frame()
        return True

    def run_on_video(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"❌ Не вдалося відкрити відео: {video_path}")
            return True
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        if self.any_model_uses_tracker:
            self._initialize_trackers()

        should_quit = False
        while self.cap.isOpened():
            if not self.is_paused:
                if not self._process_single_frame():
                    break
            else:
                self._draw_frame()

            key = cv2.waitKeyEx(20)      
                 
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                should_quit = True
            else:
                should_quit, should_next_video = self._handle_key_press(key)

            if should_quit or should_next_video:
                break
        
        self.cap.release()
        return should_quit

    def on_trackbar_change(self, trackbar_value):
        if self.is_paused and self.cap is not None and self.cap.isOpened():
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                target_frame_num = int((trackbar_value / 1000.0) * total_frames)
                if abs(target_frame_num - self.cap.get(cv2.CAP_PROP_POS_FRAMES)) > 1:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
                    self._process_single_frame()