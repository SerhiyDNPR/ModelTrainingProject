# YOLOv9Wrapper.py

from ultralytics import YOLO
from Validators.model_wrapper import ModelWrapper
from Validators.prediction import Prediction

class YOLOv9Wrapper(ModelWrapper):
    """Обгортка для моделей YOLOv9.""" 

    def load(self, model_path):
        """
        Завантажує модель YOLOv9 з вказаного файлу.
        """
        try:
            # Логіка завантаження та сама, оскільки 'ultralytics' обробляє це уніфіковано
            self.model = YOLO(model_path)
            # Переконуємось, що імена класів в обгортці відповідають іменам в моделі
            self.class_names = self.model.names
            print(f"✅ Модель YOLOv9 '{model_path}' успішно завантажена.")
        except Exception as e:
            print(f"❌ Помилка завантаження моделі YOLOv9: {e}") 
            raise

    def predict(self, frame, conf_threshold):
        """
        Виконує детекцію та трекінг об'єктів на кадрі.
        Цей метод повністю ідентичний тому, що був для YOLOv8.
        """
        predictions = []
        # Використовуємо .track() для отримання ID об'єктів
        results = self.model.track(frame, persist=True, conf=conf_threshold, verbose=False)
        
        if results and results[0].boxes:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                score = box.conf[0].item()
                class_id = int(box.cls[0].item())
                # Отримання ID для трекінгу
                track_id = int(box.id[0].item()) if box.id is not None else None
                class_name = self.class_names.get(class_id, f"Unknown_{class_id}")
                
                predictions.append(Prediction(xyxy, score, class_id, class_name, track_id))
                
        return predictions