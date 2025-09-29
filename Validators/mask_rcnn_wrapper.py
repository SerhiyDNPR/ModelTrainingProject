# mask_rcnn_wrapper.py

import torch
import torchvision
from torchvision.transforms import functional as F
from Validators.model_wrapper import ModelWrapper
from Validators.prediction import Prediction

class MaskRCNNWrapper(ModelWrapper):
    """Обгортка для моделей Mask R-CNN (ResNet50 FPN)."""

    def load(self, model_path):
        """
        Завантажує модель Mask R-CNN та адаптує її класифікатор.
        """
        try:
            # Для моделей на базі R-CNN кількість класів включає фон
            num_classes_with_bg = len(self.class_names) + 1
            
            # Створюємо архітектуру моделі Mask R-CNN
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)

            # Замінюємо "голову" класифікатора bounding box
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes_with_bg)
            
            # --- Примітка: "голова" для масок не змінюється, оскільки fine-tuning
            # зазвичай фокусується на класифікаторі. За потреби можна замінити і її.
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Перевіряємо, чи ваги збережені у словнику
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict)
            
            self.model = model.to(self.device).eval()
            print(f"✅ Модель Mask R-CNN '{model_path}' успішно завантажена.")
        except Exception as e:
            print(f"❌ Помилка завантаження моделі Mask R-CNN: {e}")
            raise

    def predict(self, frame, conf_threshold):
        """
        Робить передбачення на одному кадрі.
        Повертає список об'єктів Prediction, ігноруючи маски сегментації.
        """
        predictions = []
        
        # Конвертація BGR (OpenCV) -> RGB -> Tensor
        rgb_frame = frame[:, :, ::-1].copy()
        tensor_frame = F.to_tensor(rgb_frame).to(self.device)
        
        with torch.no_grad():
            results = self.model([tensor_frame])[0]

        # Обробляємо результати так само, як і для Faster R-CNN
        for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
            if score.item() >= conf_threshold:
                # Важливо: Torchvision включає фон як клас 0, тому віднімаємо 1
                class_id = label.item() - 1
                if 0 <= class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    predictions.append(
                        Prediction(
                            box.cpu().numpy(),
                            score.item(),
                            class_id,
                            class_name
                        )
                    )
        return predictions