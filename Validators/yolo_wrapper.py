# yolo_wrapper.py

from ultralytics import YOLO
from .model_wrapper import ModelWrapper
from .prediction import Prediction
# CHANGE: Added imports for SAHI
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
from utils.sahi_result_adapter import SahiResultAdapter


class YOLOWrapper(ModelWrapper):
    """Обгортка для моделей YOLOv8 з опціональною підтримкою SAHI."""

    # CHANGE: Added use_sahi parameter to the load method
    def load(self, model_path, use_sahi=False):
        self.use_sahi = use_sahi
        try:
            if self.use_sahi:
                print("✨ SAHI slicing is ENABLED. Loading model via SAHI's AutoDetectionModel...")
                self.model = AutoDetectionModel.from_pretrained(
                    model_type='yolov8',
                    model_path=model_path,
                    device=self.device,
                )
                self.class_names = self.model.model.names
            else:
                print("✨ SAHI slicing is DISABLED. Loading model via Ultralytics YOLO...")
                self.model = YOLO(model_path)
                self.class_names = self.model.names
            
            print(f"✅ Модель YOLOv8 '{model_path}' успішно завантажена.")

        except Exception as e:
            print(f"❌ Помилка завантаження моделі YOLOv8: {e}")
            raise

    def predict(self, frame, conf_threshold):
        # CHANGE: Added conditional logic for SAHI prediction
        if hasattr(self, 'use_sahi') and self.use_sahi:
            self.model.confidence_threshold = conf_threshold
            
            result = get_sliced_prediction(
                frame,
                detection_model=self.model,
                slice_height=SahiResultAdapter.SAHI_SETTINGS['SLICE_HEIGHT'],
                slice_width=SahiResultAdapter.SAHI_SETTINGS['SLICE_WIDTH'],
                overlap_height_ratio=SahiResultAdapter.SAHI_SETTINGS['OVERLAP_HEIGHT_RATIO'],
                overlap_width_ratio=SahiResultAdapter.SAHI_SETTINGS['OVERLAP_WIDTH_RATIO'],
            )
            
            sahi_predictions = result.object_prediction_list
            adapter = SahiResultAdapter(sahi_predictions)
            
            predictions = []
            for box in adapter.boxes:
                xyxy = box.xyxy[0] 
                score = box.conf[0]
                class_id = int(box.cls[0])
                class_name = self.class_names.get(class_id, f"Unknown_{class_id}")
                
                # SAHI does not support tracking out-of-the-box, so track_id is None
                predictions.append(Prediction(xyxy, score, class_id, class_name, None))
            return predictions

        else:
            # Original YOLOv8 tracking logic
            predictions = []
            results = self.model.track(frame, persist=True, conf=conf_threshold, verbose=False)
            
            if results and results[0].boxes:
                for box in results[0].boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    score = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    track_id = int(box.id[0].item()) if box.id is not None else None
                    class_name = self.class_names.get(class_id, f"Unknown_{class_id}")
                    
                    predictions.append(Prediction(xyxy, score, class_id, class_name, track_id))
                    
            return predictions