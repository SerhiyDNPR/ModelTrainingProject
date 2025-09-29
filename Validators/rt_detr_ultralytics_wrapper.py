# rt_detr_ultralytics_wrapper.py

from ultralytics import RTDETR # Імпортуємо RTDETR з ultralytics
from Validators.model_wrapper import ModelWrapper
from Validators.prediction import Prediction

class RTDETRUltralyticsWrapper(ModelWrapper):
    """Обгортка для моделей RT-DETR, навчених через Ultralytics."""

    def __init__(self, class_names, device):
        # Ultralytics сам керує пристроєм, але ми залишаємо параметр для сумісності
        super().__init__(class_names, device)

    def load(self, model_path):
        """
        Завантажує навчену модель RT-DETR з файлу .pt.
        """
        try:
            # Ініціалізуємо модель, передаючи шлях до навчених ваг
            self.model = RTDETR(model_path)
            print(f"✅ Модель RT-DETR (Ultralytics) '{model_path}' успішно завантажена.")
        except Exception as e:
            print(f"❌ Помилка завантаження моделі RT-DETR (Ultralytics): {e}")
            raise

    def predict(self, frame, conf_threshold):
        """
        Робить передбачення на одному кадрі.
        """
        if self.model is None:
            raise RuntimeError("Модель не завантажена. Викличте метод load() перед predict().")

        predictions = []
        
        # Виконуємо передбачення. verbose=False, щоб уникнути зайвого виводу в консоль.
        results = self.model.predict(frame, conf=conf_threshold, verbose=False)
        
        # Результат вже містить оброблені дані, їх потрібно лише розпарсити
        # results[0] містить результат для першого зображення (ми передаємо лише одне)
        result = results[0]
        boxes = result.boxes
        
        for box in boxes:
            # Отримуємо координати у форматі [xmin, ymin, xmax, ymax]
            xyxy = box.xyxy[0].cpu().numpy()
            
            # Отримуємо впевненість та ID класу
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())
            
            class_name = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else "Unknown"
            
            predictions.append(
                Prediction(
                    xyxy,
                    conf,
                    class_id,
                    class_name
                )
            )
            
        return predictions