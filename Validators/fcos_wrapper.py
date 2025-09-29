# fcos_wrapper.py

import torch
import torchvision
from torchvision.transforms import functional as F
from Validators.model_wrapper import ModelWrapper
from Validators.prediction import Prediction

class FCOSWrapper(ModelWrapper):
    """Обгортка для моделей FCOS (ResNet50 FPN)."""

    def load(self, model_path):
        try:
            num_classes = len(self.class_names)
            
            # 1. Створюємо архітектуру FCOS
            model = torchvision.models.detection.fcos_resnet50_fpn(weights=torchvision.models.detection.FCOS_ResNet50_FPN_Weights.DEFAULT)

            # 2. Отримуємо параметри для заміни "голови"
            in_channels = model.head.classification_head.conv[0].in_channels
            num_anchors = model.head.classification_head.num_anchors

            # 3. Замінюємо фінальний згортковий шар
            model.head.classification_head.cls_logits = torch.nn.Conv2d(
                in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
            )
            
            # 4. Оновлюємо кількість класів в самій голові
            model.head.classification_head.num_classes = num_classes
            
            # 5. Завантажуємо навчені ваги
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict)
            
            self.model = model.to(self.device).eval()
            print(f"✅ Модель FCOS '{model_path}' успішно завантажена.")
        except Exception as e:
            print(f"❌ Помилка завантаження моделі FCOS: {e}")
            raise

    def predict(self, frame, conf_threshold):
        predictions = []
        
        # Конвертація BGR (OpenCV) -> RGB -> Tensor
        rgb_frame = frame[:, :, ::-1].copy()
        tensor_frame = F.to_tensor(rgb_frame).to(self.device)
        
        with torch.no_grad():
            results = self.model([tensor_frame])[0]

        for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
            if score.item() >= conf_threshold:
                # Важливо: FCOS, на відміну від Faster R-CNN, повертає 0-індексовані мітки,
                # тому віднімати 1 не потрібно.
                class_id = label.item()
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