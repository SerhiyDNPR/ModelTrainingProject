# utils/backbone_factory.py

import torch
from torch import nn
from torch.nn import functional as F_torch
import torchvision.models as models
from torchvision.models.detection.backbone_utils import BackboneWithFPN, resnet_fpn_backbone

try:
    import timm
except ImportError:
    timm = None

# --- Функція для створення сумісного субкласу ---
def create_compatible_timm_backbone(timm_model: nn.Module, out_names: list[str]) -> nn.Module:
    
    class CompatibleTimmBackbone(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self._out_features = out_names 

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            features = self.model(x)
            out = {}
            for i, name in enumerate(self._out_features):
                out[name] = features[i]
            return out

    return CompatibleTimmBackbone(timm_model)


def create_fpn_backbone(backbone_type: str, pretrained: bool = True, input_img_size=None):
    """
    Створює backbone з FPN, включаючи ResNet, EfficientNet та Swin Transformer.
    Приймає input_img_size для коректної ініціалізації timm моделей.
    """
    if timm is None:
        if 'efficientnet' in backbone_type or 'swin' in backbone_type:
             raise ImportError("Для використання EfficientNet або Swin Transformer потрібна бібліотека 'timm'.")

    out_channels = 256
    timm_backbone_base = None
    
    # ------------------------------------------------------------------
    # --- ЛОГІКА ДЛЯ TIMM BACKBONES (EfficientNet та Swin) ---
    # ------------------------------------------------------------------
    if 'efficientnet' in backbone_type or 'swin' in backbone_type:
        
        timm_model_out_indices = None
        timm_args = {}
        
        if 'efficientnet' in backbone_type:
            timm_model_out_indices = (2, 3, 4) 
            # EfficientNet НЕ ПРИЙМАЄ img_size при features_only=True
            
        elif 'swin' in backbone_type:
            timm_model_out_indices = (1, 2, 3) 
            
            if isinstance(input_img_size, int):
                size_for_timm = (input_img_size, input_img_size)
            elif isinstance(input_img_size, tuple) and len(input_img_size) == 2:
                size_for_timm = input_img_size
            else:
                size_for_timm = (800, 800) 
            
            timm_args = {'img_size': size_for_timm} # Swin вимагає img_size


        timm_model_base = timm.create_model(
            backbone_type,
            features_only=True,
            out_indices=timm_model_out_indices, 
            pretrained=pretrained,
            **timm_args
        )
        
        in_channels_list = timm_model_base.feature_info.channels()
        timm_model_out_names = ['0', '1', '2']

        # 1. Обгортка для створення іменованих виходів
        compatible_backbone_base = create_compatible_timm_backbone(timm_model_base, timm_model_out_names)
        
        # 2. Створення FPN (P3-P5)
        backbone = BackboneWithFPN(
            compatible_backbone_base,
            return_layers={name: name for name in timm_model_out_names},
            in_channels_list=in_channels_list,
            out_channels=out_channels,
        )
        
        return backbone
        
    # ------------------------------------------------------------------
    # --- ЛОГІКА ДЛЯ RESNET50 (Створення стандартної torchvision функції) ---
    # ------------------------------------------------------------------
    elif backbone_type == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        
        backbone = resnet_fpn_backbone(
            'resnet50',
            weights=weights,
            trainable_layers=5
        )
        return backbone
        
    else:
        raise ValueError(f"Непідтримуваний тип backbone: {backbone_type}")