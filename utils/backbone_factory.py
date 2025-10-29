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
    """
    Створює спеціалізований субклас nn.Module, який інкапсулює модель timm, 
    повертає іменований словник і має атрибут _out_features для BackboneWithFPN.
    """
    
    class CompatibleTimmBackbone(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            # Встановлюємо атрибут _out_features для сумісності з BackboneWithFPN
            self._out_features = out_names 

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            # Викликає forward-метод timm моделі (який повертає список тензорів)
            features = self.model(x)
            out = {}
            for i, name in enumerate(self._out_features):
                out[name] = features[i]
            return out

    return CompatibleTimmBackbone(timm_model)

# ------------------------------------------------------------------
# --- НОВА КАСТОМНА FPN (Заміна BackboneWithFPN) ---
# ------------------------------------------------------------------
class CustomFPN(nn.Module):
    """
    Кастомна FPN-реалізація для використання з timm-бекбонами.
    Приймає список карт ознак і обробляє їх.
    """
    def __init__(self, timm_model: nn.Module, in_channels_list: list[int], out_channels: int):
        super().__init__()
        self.timm_model = timm_model
        
        self.out_channels = out_channels 
        
        # Бічні та верхні (латеральні та доповнюючі) шари
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        self.last_level = LastLevelP6P7(out_channels, out_channels)

        for in_channels in in_channels_list:
            # Тут in_channels має бути 768 для Swin C5
            self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, 1))
            self.layer_blocks.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

        # Імена виходів для сумісності з torchvision детектором (RetinaNet/FasterRCNN)
        self._out_features = [str(i) for i in range(len(in_channels_list) + 2)] # P3, P4, P5, P6, P7

        # Ініціалізація ваг
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        
        # Отримати карти ознак від timm моделі (список тензорів)
        timm_features = self.timm_model(x) 
        
        # --- КРИТИЧНЕ ВИПРАВЛЕННЯ: Транспонування (NHWC -> NCHW) ---
        # Виходи Swin/ViT часто мають розмірність (N, H, W, C). Conv2d очікує (N, C, H, W).
        def safe_permute(f):
            # Якщо тензор має 4 розмірності і кількість каналів (f.shape[-1])
            # відповідає очікуваній кількості каналів FPN (self.inner_blocks[-1].in_channels), 
            # транспортуємо його.
            if f.dim() == 4 and f.shape[-1] == self.inner_blocks[-1].in_channels:
                 return f.permute(0, 3, 1, 2) # N, H, W, C -> N, C, H, W
            return f

        timm_features = [safe_permute(f) for f in timm_features]
        # ----------------------------------------------------------------------
        
        last_inner = self.inner_blocks[-1](timm_features[-1])
        results = [self.layer_blocks[-1](last_inner)]

        for feature, inner_block, layer_block in zip(
            timm_features[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not last_inner.shape[-2:] == feature.shape[-2:]:
                last_inner = F_torch.interpolate(last_inner, size=feature.shape[-2:], mode="nearest")
            
            inner_top_down = F_torch.interpolate(last_inner, size=feature.shape[-2:], mode="nearest")
            inner_lateral = inner_block(feature)
            last_inner = inner_lateral + inner_top_down
            
            results.insert(0, layer_block(last_inner))
            
        # P6 та P7 (використовуємо P5, який є останнім елементом результатів)
        p6, p7 = self.last_level(results[-1])
        results.append(p6)
        results.append(p7)

        # Форматуємо вихід у словник
        out = {}
        for name, feature in zip(self._out_features, results):
            out[name] = feature
        
        return out

# --- Додаткові рівні P6 та P7 для FPN ---
class LastLevelP6P7(nn.Module):
    """
    Цей модуль додає додаткові рівні P6 та P7 до Feature Pyramid Network (FPN),
    використовуючи вихід з останнього рівня FPN (P5).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        
    def forward(self, p5: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p6 = self.p6(p5)
        p7 = self.p7(F_torch.relu(p6))
        return p6, p7

def create_fpn_backbone(backbone_type: str, pretrained: bool = True):
    """
    Створює backbone з FPN, включаючи ResNet, EfficientNet та Swin Transformer.
    """
    if timm is None:
        if 'efficientnet' in backbone_type or 'swin' in backbone_type:
             raise ImportError("Для використання EfficientNet або Swin Transformer потрібна бібліотека 'timm'.")

    # --- Загальні параметри FPN ---
    out_channels = 256
    
    # ------------------------------------------------------------------
    # --- ЛОГІКА ДЛЯ TIMM BACKBONES (EfficientNet та Swin) ---
    # ------------------------------------------------------------------
    if 'efficientnet' in backbone_type or 'swin' in backbone_type:
        
        if 'efficientnet' in backbone_type:
            timm_model_out_indices = (2, 3, 4) 
        else: # Swin
            timm_model_out_indices = (1, 2, 3) 
            
        timm_model_base = timm.create_model(
            backbone_type,
            features_only=True,
            out_indices=timm_model_out_indices, 
            pretrained=pretrained,
            img_size=800 
        )
        
        in_channels_list = timm_model_base.feature_info.channels()
        
        # Використовуємо кастомну FPN для обходу проблеми сумісності
        backbone = CustomFPN(timm_model_base, in_channels_list, out_channels)
        return backbone
        
    # ------------------------------------------------------------------
    # --- ЛОГІКА ДЛЯ RESNET50 (Використовуємо стандартну torchvision функцію) ---
    # ------------------------------------------------------------------
    elif backbone_type == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        
        # ResNet використовує стандартний BackboneWithFPN, який коректно імпортовано
        backbone = resnet_fpn_backbone(
            'resnet50',
            weights=weights,
            trainable_layers=5
        )
        return backbone
        
    else:
        raise ValueError(f"Непідтримуваний тип backbone: {backbone_type}")