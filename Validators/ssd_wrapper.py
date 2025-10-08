# Validators/ssd_wrapper.py

import sys
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.ops import Conv2dNormActivation
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from .model_wrapper import ModelWrapper
from .prediction import Prediction

class SSDWrapper(ModelWrapper):
    """
    Універсальна обгортка для моделей SSD з опціональною підтримкою SAHI.
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

    def load(self, model_path, use_sahi=False):
        self.use_sahi = use_sahi
        try:
            backbone_type = self._select_backbone()
            num_classes = len(self.class_names)
            
            if self.use_sahi:
                print("✨ SAHI slicing ENABLED. Завантаження моделі через AutoDetectionModel...")
                self.model = AutoDetectionModel.from_pretrained(
                    model_type='torchvision', # SAHI має вбудовану підтримку torchvision
                    model_path=model_path,
                    config_path=backbone_type, # Передаємо тип бекбону як "конфігурацію"
                    num_classes=num_classes,
                    device=self.device,
                )
            else:
                print("✨ SAHI slicing DISABLED. Стандартне завантаження моделі...")
                if backbone_type == 'vgg16':
                    model_instance = torchvision.models.detection.ssd300_vgg16(num_classes=num_classes + 1)
                else: # mobilenet
                    model_instance = torchvision.models.detection.ssdlite320_mobilenet_v3_large(num_classes=num_classes + 1)
                
                state_dict = torch.load(model_path, map_location=self.device).get('model_state_dict', torch.load(model_path, map_location=self.device))
                model_instance.load_state_dict(state_dict)
                self.model = model_instance.to(self.device).eval()

            print(f"✅ Модель SSD ({backbone_type.upper()}) '{model_path}' успішно завантажена.")
        except Exception as e:
            print(f"❌ Помилка завантаження моделі SSD: {e}")
            raise

    def predict(self, frame, conf_threshold):
        if hasattr(self, 'use_sahi') and self.use_sahi:
            # Логіка передбачення з нарізкою SAHI
            self.model.confidence_threshold = conf_threshold
            
            result = get_sliced_prediction(
                frame,
                detection_model=self.model,
                slice_height=512,
                slice_width=512,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )
            
            predictions = []
            for pred in result.object_prediction_list:
                box = pred.bbox.to_xyxy()
                score = pred.score.value
                class_id = pred.category.id
                class_name = pred.category.name
                
                predictions.append(
                    Prediction(box, score, class_id, class_name, track_id=None)
                )
            return predictions

        else:
            # Стандартна логіка передбачення без нарізки
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
                            Prediction(box.cpu().numpy(), score.item(), class_id, class_name, track_id=None)
                        )
            return predictions