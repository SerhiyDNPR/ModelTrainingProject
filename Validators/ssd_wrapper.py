# Validators/ssd_wrapper.py

import torch
import os
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSDClassificationHead, SSDRegressionHead
from torchvision.ops import Conv2dNormActivation
# ---------------------------------------------------------------
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from .model_wrapper import ModelWrapper
from .prediction import Prediction

class SSDWrapper(ModelWrapper):
    """
    Універсальна обгортка для моделей SSD з опціональною підтримкою SAHI.
    Адаптовано для завантаження моделей, навчених за допомогою SSDTrainer.
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

    def _build_model(self, backbone_type, num_classes):
        """
        Створює екземпляр моделі з правильною архітектурою "голови",
        аналогічно до того, як це робиться в SSDTrainer.
        """
        print("🔧 Створення архітектури моделі для завантаження ваг...")
        if backbone_type == 'vgg16':
            model = torchvision.models.detection.ssd300_vgg16(weights=None, num_classes=num_classes)
        else: # mobilenet
            model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=num_classes)

        # --- ВАЖЛИВО: Генератор якорів має бути ідентичним тому, що використовувався при навчанні ---
        model.anchor_generator = DefaultBoxGenerator(
            [
                [0.045, 0.07, 0.1], [0.1, 0.18, 0.25], [0.25, 0.4, 0.5],
                [0.5, 0.6, 0.7], [0.7, 0.8, 0.9], [0.9, 0.93, 0.95] 
            ]
        )
        
        in_channels = []
        for layer in model.head.classification_head.module_list:
            if isinstance(layer, torch.nn.Sequential) and isinstance(layer[0], Conv2dNormActivation):
                in_channels.append(layer[0][0].in_channels)
            else:
                in_channels.append(layer.in_channels)
        
        num_anchors = model.anchor_generator.num_anchors_per_location()
        
        model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
        model.head.regression_head = SSDRegressionHead(in_channels, num_anchors)
        
        return model

    def load(self, model_path, use_sahi=False):
        self.use_sahi = use_sahi
        try:
            backbone_type = self._select_backbone()
            num_classes_for_head = len(self.class_names) + 1
            
            # 1. Створюємо архітектуру моделі
            model_instance = self._build_model(backbone_type, num_classes_for_head)
            
            # 2. Завантажуємо ваги з чекпоінта
            state_dict = torch.load(model_path, map_location=self.device).get('model_state_dict', torch.load(model_path, map_location=self.device))
            model_instance.load_state_dict(state_dict)
            model_instance = model_instance.to(self.device).eval()

            # --- ОСЬ ТУТ БУЛА ПОМИЛКА ---
            if self.use_sahi:
                print("✨ SAHI slicing ENABLED. Ініціалізація AutoDetectionModel з готовою моделлю...")
                # ПРАВИЛЬНО: передаємо готовий об'єкт моделі, а не шлях
                self.model = AutoDetectionModel.from_pretrained(
                    model_type='torchvision',  # <--- КЛЮЧОВИЙ ДОДАНИЙ РЯДОК
                    model=model_instance,
                    category_mapping={i: name for i, name in enumerate(self.class_names)},
                    device=self.device,
                )
            else:
                print("✨ SAHI slicing DISABLED. Використовується стандартна модель.")
                self.model = model_instance

            print(f"✅ Модель SSD ({backbone_type.upper()}) '{os.path.basename(model_path)}' успішно завантажена.")
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