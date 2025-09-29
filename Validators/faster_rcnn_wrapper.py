# faster_rcnn_wrapper.py

import sys
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from Validators.model_wrapper import ModelWrapper
from Validators.prediction import Prediction


class FasterRCNNWrapper(ModelWrapper):
    """Універсальна обгортка для моделей Faster R-CNN з можливістю вибору backbone."""

    def _select_backbone(self):
        """
        Відображає меню вибору backbone і повертає вибір користувача.
        """
        print("\nБудь ласка, оберіть 'хребет' (backbone) для моделі Faster R-CNN, що завантажується:")
        print("  1: ResNet-50")
        print("  2: ResNet-101")
        print("  3: MobileNetV3-Large")
        
        while True:
            choice = input("Ваш вибір (1, 2 або 3): ").strip()
            if choice == '1':
                print("✅ Обрано архітектуру на базі ResNet-50.")
                return 'resnet50'
            elif choice == '2':
                print("✅ Обрано архітектуру на базі ResNet-101.")
                return 'resnet101'
            elif choice == '3':
                print("✅ Обрано архітектуру на базі MobileNetV3-Large.")
                return 'mobilenet'
            else:
                print("❌ Невірний вибір. Будь ласка, введіть 1, 2 або 3.")

    def load(self, model_path):
        try:
            backbone_type = self._select_backbone()

            num_classes_with_bg = len(self.class_names) + 1
            
            # Створюємо порожню модель з відповідним backbone
            if backbone_type == 'resnet50':
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
            elif backbone_type == 'resnet101':
                try:
                    from torchvision.models import ResNet101_Weights
                    backbone = resnet_fpn_backbone('resnet101', weights=ResNet101_Weights.IMAGENET1K_V1)
                except (ImportError, AttributeError):
                    print("⚠️ Попередження: не вдалося завантажити ваги за новим API. Використовується 'pretrained=True'.")
                    backbone = resnet_fpn_backbone('resnet101', pretrained=True)
                model = FasterRCNN(backbone, num_classes=91)
            
            elif backbone_type == 'mobilenet':
                 model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
            else:
                print(f"❌ Помилка: невідомий тип backbone '{backbone_type}'.")
                sys.exit(1)

            # Замінюємо голову моделі на потрібну кількість класів
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes_with_bg)
            
            # Завантажуємо ваги з файлу
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
            
            self.model = model.to(self.device).eval()
            print(f"✅ Модель Faster R-CNN ({backbone_type.upper()}) '{model_path}' успішно завантажена.")
        except Exception as e:
            print(f"❌ Помилка завантаження моделі Faster R-CNN: {e}")
            raise

    def predict(self, frame, conf_threshold):
        predictions = []
        rgb_frame = frame[:, :, ::-1].copy()
        tensor_frame = F.to_tensor(rgb_frame).to(self.device)
        
        with torch.no_grad():
            results = self.model([tensor_frame])[0]

        for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
            if score.item() >= conf_threshold:
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