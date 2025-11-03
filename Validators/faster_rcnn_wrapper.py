# faster_rcnn_wrapper.py

import sys
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from Validators.model_wrapper import ModelWrapper
from Validators.prediction import Prediction
from trainers.backbone_factory import create_fpn_backbone 
import cv2 

BACKBONE_CHOICES = {
    '1': ('resnet50', (800, 800), "ResNet-50"),
    '2': ('resnet101', (800, 800), "ResNet-101"),
    '3': ('mobilenet', (640, 640), "MobileNetV3-Large"),
    '4': ('swin_tiny_patch4_window7_224', (800, 800), "Swin-T (Tiny Transformer)"),
    '5': ('swin_small_patch4_window7_224', (1024, 1024), "Swin-S (Small Transformer)"),
}

class FasterRCNNWrapper(ModelWrapper):
    """Універсальна обгортка для моделей Faster R-CNN з можливістю вибору backbone."""

    def __init__(self, class_names, device):
        super().__init__(class_names, device)
        self.input_size = (0, 0)
        self.backbone_type = None

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
                self.backbone_type = backbone_type # Зберігаємо для predict
                print(f"✅ Обрано архітектуру на базі: {desc.split(' (')[0]}")
                
                if 'resnet' not in backbone_type and 'mobilenet' not in backbone_type:
                    try:
                        import timm
                    except ImportError:
                        print("❌ Помилка: бібліотека 'timm' не встановлена. Оберіть інший backbone.")
                        continue

                    # --- Запит розміру ТІЛЬКИ ДЛЯ SWIN ---
                    if 'swin' in backbone_type:
                        print(f"⚠️ Увага: Рекомендований розмір для цієї моделі: {recommended_size[0]}x{recommended_size[1]}.")
                        custom_size_input = input("Введіть розмір зображення (одна сторона, наприклад, 800), на якому навчалася модель, або натисніть Enter, щоб використати рекомендований: ").strip()
                        
                        try:
                            input_size = int(custom_size_input) if custom_size_input else recommended_size[0]
                        except ValueError:
                            print("❌ Невірний формат розміру. Використовується рекомендований розмір.")
                            input_size = recommended_size[0]

                        return backbone_type, (input_size, input_size)
                    
                    else:
                        # Тут може бути логіка для EfficientNet, якщо його додати
                        return backbone_type, recommended_size
                
                else:
                    # Для ResNet/MobileNet повертаємо рекомендований розмір без запиту
                    return backbone_type, recommended_size
            else:
                print(f"❌ Невірний вибір. Будь ласка, введіть число від 1 до {len(BACKBONE_CHOICES)}.")

    def load(self, model_path):
        try:
            backbone_type, image_size = self._select_backbone()
            num_classes_with_bg = len(self.class_names) + 1
            model = None
            
            # Визначаємо, чи потрібен input_img_size (для Swin)
            input_size_param = image_size if 'swin' in backbone_type else None
            
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
                backbone = create_fpn_backbone(backbone_type, pretrained=False, input_img_size=input_size_param)
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
        """Робить передбачення на одному кадрі з урахуванням умовного масштабування."""
        predictions = []
        
        # 1. Зберігаємо оригінальні розміри
        H_orig, W_orig, _ = frame.shape
        rgb_frame = frame[:, :, ::-1].copy()
        
        # 2. Визначаємо, чи потрібне явне масштабування для входу. 
        is_swin_backbone = self.backbone_type and 'swin' in self.backbone_type
        
        frame_to_process = rgb_frame
        should_rescale_boxes = False
        
        W_target, H_target = self.input_size if self.input_size else (0, 0)
        
        # 3. УМОВНЕ МАШТАБУВАННЯ
        if is_swin_backbone:
            # Масштабування виконується ЛИШЕ для Swin
            if H_orig != H_target or W_orig != W_target:
                 frame_to_process = cv2.resize(rgb_frame, (W_target, H_target), interpolation=cv2.INTER_LINEAR)
                 should_rescale_boxes = True
        
        tensor_frame = F.to_tensor(frame_to_process).to(self.device)
        
        with torch.no_grad():
            results = self.model([tensor_frame])[0]

        # 4. ЗВОРОТНЕ МАШТАБУВАННЯ КООРДИНАТ
        boxes = results["boxes"].cpu().numpy()
        
        if should_rescale_boxes:
            # Коефіцієнти масштабування: (Оригінальний розмір / Цільовий розмір)
            scale_x = W_orig / W_target
            scale_y = H_orig / H_target
            
            # Масштабуємо координати: xmin, ymin, xmax, ymax
            boxes[:, [0, 2]] *= scale_x 
            boxes[:, [1, 3]] *= scale_y 
            
        # 5. Формування результатів
        for box, label, score in zip(boxes, results["labels"].cpu(), results["scores"].cpu()):
            if score.item() >= conf_threshold:
                # Faster R-CNN/Mask R-CNN використовує 1-base індексацію (0 - фон)
                class_id = label.item() - 1 
                if 0 <= class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    predictions.append(
                        Prediction(
                            box,
                            score.item(),
                            class_id,
                            class_name
                        )
                    )
        return predictions