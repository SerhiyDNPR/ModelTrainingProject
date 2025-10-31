# Validators/retinanet_wrapper.py

import os
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
from Validators.model_wrapper import ModelWrapper
from Validators.prediction import Prediction
import cv2 

from utils.backbone_factory import create_fpn_backbone

# Словник з конфігураціями backbone: назва, рекомендований розмір (W, H) та опис
BACKBONE_CONFIGS = {
    '1': ('resnet50', (800, 800), "ResNet-50 (стандартний)"),
    '2': ('tf_efficientnet_b0', (512, 512), "EfficientDet-D0 (найлегший)"),
    '3': ('tf_efficientnet_b1', (640, 640), "EfficientDet-D1"),
    '4': ('tf_efficientnet_b2', (768, 768), "EfficientDet-D2"),
    '5': ('tf_efficientnet_b3', (896, 896), "EfficientDet-D3"),
    '6': ('tf_efficientnet_b4', (1024, 1024), "EfficientDet-D4"),
    '7': ('tf_efficientnet_b5', (1280, 1280), "EfficientDet-D5"),
    '8': ('swin_tiny_patch4_window7_224', (800, 800), "Swin-T (Tiny Transformer)"),
    '9': ('swin_small_patch4_window7_224', (1024, 1024), "Swin-S (Small Transformer)"),
}

class RetinaNetWrapper(ModelWrapper):
    """Обгортка для моделей RetinaNet з можливістю вибору backbone (ResNet50 або EfficientNet/Swin)."""

    def __init__(self, class_names, device):
        super().__init__(class_names, device)
        self.input_size = (0, 0)
        self.backbone_type = None

    def _select_backbone(self):
        """
        Відображає меню вибору backbone і повертає вибір користувача та фактичний розмір.
        """
        print("\nБудь ласка, оберіть 'хребет' (backbone) для моделі RetinaNet, що завантажується:")
        for key, (_, _, description) in BACKBONE_CONFIGS.items():
            print(f"  {key}: {description}")
        
        while True:
            choice = input(f"Ваш вибір (1-{len(BACKBONE_CONFIGS)}): ").strip()
            if choice in BACKBONE_CONFIGS:
                backbone_type, recommended_size, desc = BACKBONE_CONFIGS[choice]
                self.backbone_type = backbone_type # Зберігаємо для predict
                print(f"✅ Обрано архітектуру на базі: {desc.split(' (')[0]}")
                
                if 'resnet' not in backbone_type:
                    try:
                        import timm
                    except ImportError:
                        print("❌ Помилка: бібліотека 'timm' не встановлена. Оберіть інший backbone.")
                        continue
                
                    # --- Запит розміру ТІЛЬКИ ДЛЯ TIMM МОДЕЛЕЙ ---
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
                        # Для EfficientNet повертаємо рекомендований розмір без запиту
                        return backbone_type, recommended_size
                
                else:
                    # Для ResNet повертаємо рекомендований розмір без запиту
                    return backbone_type, recommended_size
            else:
                print(f"❌ Невірний вибір. Будь ласка, введіть число від 1 до {len(BACKBONE_CONFIGS)}.")

    def load(self, model_path):
        """Завантажує модель RetinaNet та адаптує її класифікатор."""
        try:
            backbone_type, image_size = self._select_backbone() 
            num_classes = len(self.class_names)
            model = None
            in_channels = 256
            
            # --- ЛОГІКА СТВОРЕННЯ МОДЕЛІ ---
            if backbone_type == 'resnet50':
                model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights=None)
                anchor_generator = model.anchor_generator
                num_anchors = anchor_generator.num_anchors_per_location()[0]
                in_channels = model.backbone.out_channels
                
            else:
                # ЛОГІКА ДЛЯ TIMM BACKBONES (Swin/EfficientNet)
                input_size_param = image_size 
                
                if 'swin' in backbone_type:
                    backbone = create_fpn_backbone(backbone_type, pretrained=False, input_img_size=input_size_param)
                else:
                    backbone = create_fpn_backbone(backbone_type, pretrained=False) 

                anchor_sizes = tuple(
                    (x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512]
                )
                anchor_aspect_ratios = tuple([(0.5, 1.0, 2.0)] * len(anchor_sizes))
                anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=anchor_aspect_ratios)
                
                in_channels = backbone.out_channels
                num_anchors = anchor_generator.num_anchors_per_location()[0]
                
                model = torchvision.models.detection.RetinaNet(
                    backbone, num_classes=num_classes, anchor_generator=anchor_generator
                )

            # --- Адаптація голови ---
            new_head = RetinaNetHead(
                in_channels=in_channels,
                num_anchors=num_anchors,
                num_classes=num_classes
            )
            model.head = new_head
            
            # --- Завантаження ваг ---
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)
            
            self.model = model.to(self.device).eval()
            self.input_size = image_size
            print(f"✅ Модель RetinaNet ({backbone_type}) '{os.path.basename(model_path)}' успішно завантажена.")

        except Exception as e:
            print(f"❌ Помилка завантаження моделі RetinaNet: {e}")
            raise

    def predict(self, frame, conf_threshold):
        """Робить передбачення на одному кадрі з урахуванням умовного масштабування."""
        predictions = []
        
        # 1. Зберігаємо оригінальні розміри
        H_orig, W_orig, _ = frame.shape
        rgb_frame = frame[:, :, ::-1].copy()
        
        # 2. Визначаємо, чи потрібне явне масштабування для входу. 
        # Масштабування потрібне ЛИШЕ для Swin/EfficientNet (timm-based).
        is_fixed_size_backbone = self.backbone_type and ('swin' in self.backbone_type or 'efficientnet' in self.backbone_type)
        
        frame_to_process = rgb_frame
        should_rescale_boxes = False
        
        W_target, H_target = self.input_size 
        
        # 3. УМОВНЕ МАШТАБУВАННЯ
        if is_fixed_size_backbone:
            # Масштабування виконується ЛИШЕ для Swin/EfficientNet
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
                class_id = label.item()
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