# utils/backbone_factory.py

import torch
from torch import nn
from torch.nn import functional as F_torch
import torchvision.models as models
from torchvision.models.detection.backbone_utils import BackboneWithFPN

try:
    import timm
except ImportError:
    timm = None

# Визначаємо клас LastLevelP6P7 безпосередньо у файлі, щоб уникнути проблем з імпортом
class LastLevelP6P7(nn.Module):
    """
    Цей модуль додає додаткові рівні P6 та P7 до Feature Pyramid Network (FPN),
    використовуючи вихід з останнього рівня FPN (P5).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)

    def forward(self, p5: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p6 = self.p6(p5)
        p7 = self.p7(F_torch.relu(p6))
        return p6, p7

def create_fpn_backbone(backbone_type: str, pretrained: bool = True):
    """
    Створює backbone з FPN, точно відтворюючи логіку з RetinaNet_trainer.py.
    """
    if 'efficientnet' in backbone_type:
        if timm is None:
            raise ImportError("Для використання EfficientNet потрібна бібліотека 'timm'.")
        
        backbone_timm = timm.create_model(
            backbone_type,
            features_only=True,
            out_indices=(2, 3, 4),
            pretrained=pretrained
        )
        
        backbone = BackboneWithFPN(
            backbone_timm,
            return_layers={'2': '0', '3': '1', '4': '2'},
            in_channels_list=backbone_timm.feature_info.channels(),
            out_channels=256,
            extra_blocks=LastLevelP6P7(256, 256)
        )
        return backbone
    
    elif backbone_type == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.detection.backbone_utils.resnet_fpn_backbone(
            'resnet50',
            weights=weights,
            trainable_layers=5
        )
        return backbone
        
    else:
        raise ValueError(f"Непідтримуваний тип backbone: {backbone_type}")