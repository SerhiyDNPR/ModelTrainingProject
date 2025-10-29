# faster_rcnn_wrapper.py

import sys
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from Validators.model_wrapper import ModelWrapper
from Validators.prediction import Prediction
from utils.backbone_factory import create_fpn_backbone 
import cv2 # ДОДАНО

BACKBONE_CHOICES = {
    '1': ('resnet50', (800, 800), "ResNet-50"),
    '2': ('resnet101', (800, 800), "ResNet-101"),
    '3': ('mobilenet', (640, 640), "MobileNetV3-Large"),
    '4': ('swin_tiny_patch4_window7_224', (800, 800), "Swin-T (Tiny Transformer)"),
    '5': ('swin_small_patch4_window7_224', (800, 800), "Swin-S (Small Transformer)"),
}

class FasterRCNNWrapper(ModelWrapper):
    """Універсальна обгортка для моделей Faster R-CNN з можливістю вибору backbone."""

    def _select_backbone(self):
        """
        Відображає меню вибору backbone і повертає вибір користувача та розмір.
        """
        print("\nБудь ласка, оберіть 'хребет' (backbone) для моделі Faster R-CNN, що завантажується:")
        for key, (_, _, description) in BACKBONE_CHOICES.items():
            print(f"  {key}: {description}")
        
        while True:
            choice = input(f"Ваш вибір (1-{len(BACKBONE_CHOICES)}): ").strip()
            if choice in BACKBONE_CHOICES:
                backbone_type, recommended_size, desc = BACKBONE_CHOICES[choice]
                print(f"✅ Обрано архітектуру на базі: {desc.split(' (')[0]}")
                
                if backbone_type.startswith('swin'):
                    try:
                        import timm
                    except ImportError:
                        print("❌ Помилка: бібліотека 'timm' не встановлена. Оберіть інший backbone.")
                        continue

                # Додаткове питання про розмір
                print(f"⚠️ Увага: Рекомендований розмір для цієї моделі: {recommended_size[0]}x{recommended_size[1]}.")
                custom_size_input = input("Введіть розмір зображення (одна сторона, наприклад, 800), на якому навчалася модель, або натисніть Enter, щоб використати рекомендований: ").strip()
                
                try:
                    input_size = int(custom_size_input) if custom_size_input else recommended_size[0]
                except ValueError:
                    print("❌ Невірний формат розміру. Використовується рекомендований розмір.")
                    input_size = recommended_size[0]

                return backbone_type, (input_size, input_size)
            else:
                print(f"❌ Невірний вибір. Будь ласка, введіть число від 1 до {len(BACKBONE_CHOICES)}.")

    def load(self, model_path):
        try:
            # Отримуємо backbone_type та image_size
            backbone_type, image_size = self._select_backbone()
            num_classes_with_bg = len(self.class_names) + 1
            model = None
            
            # --- ResNet & MobileNet (using torchvision built-in) ---
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
            
            # --- Swin (using custom FPN from backbone_factory) ---
            elif backbone_type.startswith('swin'):
                # Swin ВИМАГАЄ input_img_size
                backbone = create_fpn_backbone(backbone_type, pretrained=False, input_img_size=image_size)
                model = FasterRCNN(backbone, num_classes=91) 
            
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
            self.input_size = image_size # Зберігаємо розмір
            print(f"✅ Модель Faster R-CNN ({backbone_type.upper().replace('_', '-')}) '{model_path}' успішно завантажена.")
        except Exception as e:
            print(f"❌ Помилка завантаження моделі Faster R-CNN: {e}")
            raise

    def predict(self, frame, conf_threshold):
        predictions = []
        rgb_frame = frame[:, :, ::-1].copy()
        
        # --- ВИПРАВЛЕННЯ: Масштабування кадру до розміру, на якому тренувалась модель ---
        W, H = self.input_size 
        
        if frame.shape[0] != H or frame.shape[1] != W:
             resized_frame = cv2.resize(rgb_frame, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
             resized_frame = rgb_frame
             
        tensor_frame = F.to_tensor(resized_frame).to(self.device)
        # ---------------------------------------------------------------------------------

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