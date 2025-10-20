# Validators/retinanet_wrapper.py

import os
import torch
from torchvision.transforms import functional as F
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
from Validators.model_wrapper import ModelWrapper
from Validators.prediction import Prediction

from utils.backbone_factory import create_fpn_backbone

# Словник з конфігураціями backbone, аналогічний до трейнера
BACKBONE_CONFIGS = {
    '1': ('resnet50', "ResNet-50 (стандартний, збалансований)"),
    '2': ('tf_efficientnet_b0', "EfficientDet-D0 (найлегший)"),
    '3': ('tf_efficientnet_b1', "EfficientDet-D1 (кращий баланс швидкість/точність)"),
    '4': ('tf_efficientnet_b2', "EfficientDet-D2"),
    '5': ('tf_efficientnet_b3', "EfficientDet-D3"),
    '6': ('tf_efficientnet_b4', "EfficientDet-D4"),
    '7': ('tf_efficientnet_b5', "EfficientDet-D5 (вища точність, повільніший)"),
}

class RetinaNetWrapper(ModelWrapper):
    """Обгортка для моделей RetinaNet з можливістю вибору backbone (ResNet50 або EfficientNet)."""

    def _select_backbone(self):
        """Відображає меню вибору backbone і повертає вибір користувача."""
        print("\nБудь ласка, оберіть 'хребет' (backbone) для моделі RetinaNet, що завантажується:")
        for key, (_, description) in BACKBONE_CONFIGS.items():
            print(f"  {key}: {description}")
        
        while True:
            choice = input(f"Ваш вибір (1-{len(BACKBONE_CONFIGS)}): ").strip()
            if choice in BACKBONE_CONFIGS:
                backbone_type, desc = BACKBONE_CONFIGS[choice]
                print(f"✅ Обрано архітектуру на базі: {desc.split(' (')[0]}")
                if 'efficientnet' in backbone_type:
                    try:
                        import timm
                    except ImportError:
                        print("❌ Помилка: бібліотека 'timm' не встановлена. Оберіть інший backbone.")
                        continue
                return backbone_type
            else:
                print(f"❌ Невірний вибір. Будь ласка, введіть число від 1 до {len(BACKBONE_CONFIGS)}.")

    def load(self, model_path):
        """Завантажує модель RetinaNet та адаптує її класифікатор."""
        try:
            backbone_type = self._select_backbone()
            num_classes = len(self.class_names)
            
            backbone = create_fpn_backbone(backbone_type, pretrained=False)
            # -----------------------------------------------

            anchor_generator = AnchorGenerator.from_config(
                config={
                    "sizes": tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512]),
                    "aspect_ratios": tuple([(0.5, 1.0, 2.0)] * 5),
                }
            )
            head = RetinaNetHead(
                backbone.out_channels, 
                anchor_generator.num_anchors_per_location()[0], 
                num_classes
            )
            model = RetinaNet(backbone, num_classes=num_classes, anchor_generator=anchor_generator, head=head)

            # --- Загальна логіка завантаження ваг ---
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)
            
            self.model = model.to(self.device).eval()
            print(f"✅ Модель RetinaNet ({backbone_type}) '{os.path.basename(model_path)}' успішно завантажена.")

        except Exception as e:
            print(f"❌ Помилка завантаження моделі RetinaNet: {e}")
            raise

    def predict(self, frame, conf_threshold):
        """Робить передбачення на одному кадрі."""
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