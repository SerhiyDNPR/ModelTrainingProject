# retinanet_wrapper.py

import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
from Validators.model_wrapper import ModelWrapper
from Validators.prediction import Prediction

class RetinaNetWrapper(ModelWrapper):
    """Обгортка для моделей RetinaNet (ResNet50 FPN)."""

    def load(self, model_path):
        """
        Завантажує модель RetinaNet та адаптує її класифікатор.
        """
        try:
            num_classes = len(self.class_names)
            
            model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights=None)
            
            # --- КЛЮЧОВЕ ВИПРАВЛЕННЯ: Повністю відтворюємо логіку з трейнера ---

            # 1. Отримуємо параметри для нової "голови" так само, як у трейнері.
            in_channels = 256  # Відоме значення для FPN в ResNet50
            
            anchor_generator = model.anchor_generator
            if isinstance(anchor_generator, AnchorGenerator):
                num_anchors = anchor_generator.num_anchors_per_location()[0]
            else:
                # Значення за замовчуванням, якщо не вдалося визначити
                num_anchors = 9

            # 2. Створюємо екземпляр нової "голови".
            new_head = RetinaNetHead(
                in_channels=in_channels,
                num_anchors=num_anchors,
                num_classes=num_classes
            )

            # 3. Замінюємо ВСЮ голову моделі, а не її частину.
            model.head = new_head
            
            checkpoint = torch.load(model_path, map_location=self.device)

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            model.load_state_dict(state_dict)
            
            self.model = model.to(self.device).eval()
            print(f"✅ Модель RetinaNet (v2) '{model_path}' успішно завантажена.")
        except Exception as e:
            print(f"❌ Помилка завантаження моделі RetinaNet: {e}")
            raise

    def predict(self, frame, conf_threshold):
        """
        Робить передбачення на одному кадрі.
        """
        predictions = []
        
        rgb_frame = frame[:, :, ::-1].copy()
        tensor_frame = F.to_tensor(rgb_frame).to(self.device)
        
        with torch.no_grad():
            results = self.model([tensor_frame])[0]

        for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
            if score.item() >= conf_threshold:
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