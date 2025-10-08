# Validators/ssd_wrapper.py

import sys
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.ops import Conv2dNormActivation

from .model_wrapper import ModelWrapper
from .prediction import Prediction

class SSDWrapper(ModelWrapper):
    """
    Універсальна обгортка для моделей SSD, що підтримує
    бекбони VGG16 (SSD300) та MobileNetV3 (SSDLite320).
    """

    def _select_backbone(self):
        """
        Відображає меню вибору backbone і повертає вибір користувача.
        """
        print("\nБудь ласка, оберіть 'хребет' (backbone) для моделі SSD, що завантажується:")
        print("  1: VGG16 (для стандартної моделі SSD300)")
        print("  2: MobileNetV3-Large (для моделі SSDLite320)")
        
        while True:
            choice = input("Ваш вибір (1 або 2): ").strip()
            if choice == '1':
                print("✅ Обрано архітектуру на базі VGG16.")
                return 'vgg16'
            elif choice == '2':
                print("✅ Обрано архітектуру на базі MobileNetV3-Large.")
                return 'mobilenet'
            else:
                print("❌ Невірний вибір. Будь ласка, введіть 1 або 2.")

    def load(self, model_path):
        try:
            backbone_type = self._select_backbone()
            num_classes_with_bg = len(self.class_names) + 1
            
            # Створюємо порожню модель з відповідним backbone
            if backbone_type == 'vgg16':
                model = torchvision.models.detection.ssd300_vgg16(weights=None)
            elif backbone_type == 'mobilenet':
                model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=None)
            else:
                print(f"❌ Помилка: невідомий тип backbone '{backbone_type}'.")
                sys.exit(1)

            # Коректно замінюємо голову моделі для потрібної кількості класів
            # Ця логіка враховує різну структуру голів VGG та MobileNet версій
            in_channels = []
            for layer in model.head.classification_head.module_list:
                if isinstance(layer, torch.nn.Sequential) and isinstance(layer[0], Conv2dNormActivation):
                    in_channels.append(layer[0][0].in_channels)
                else:
                    in_channels.append(layer.in_channels)
            
            num_anchors = model.anchor_generator.num_anchors_per_location()
            model.head.classification_head = torchvision.models.detection.ssd.SSDClassificationHead(
                in_channels, num_anchors, num_classes_with_bg
            )
            
            # Завантажуємо ваги з файлу
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)
            
            self.model = model.to(self.device).eval()
            print(f"✅ Модель SSD ({backbone_type.upper()}) '{model_path}' успішно завантажена.")
        except Exception as e:
            print(f"❌ Помилка завантаження моделі SSD: {e}")
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
                # `label` з torchvision починається з 1 (0 - фон), тому віднімаємо 1
                class_id = label.item() - 1 
                if 0 <= class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    # Моделі з torchvision не підтримують трекінг "з коробки", тому track_id=None
                    predictions.append(
                        Prediction(
                            box.cpu().numpy(),
                            score.item(),
                            class_id,
                            class_name,
                            track_id=None 
                        )
                    )
        return predictions