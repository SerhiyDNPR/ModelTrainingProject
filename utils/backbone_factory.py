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

# ------------------------------------------------------------------
# --- НОВА КАСТОМНА FPN (Заміна BackboneWithFPN) ---
# ------------------------------------------------------------------
class CustomFPN(nn.Module):
    def __init__(self, timm_model: nn.Module, in_channels_list: list[int], out_channels: int):
        super().__init__()
        self.timm_model = timm_model
        
        self.out_channels = out_channels 
        self.in_channels_list = in_channels_list 
        
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        self.last_level = LastLevelP6P7(out_channels, out_channels)

        for in_channels in in_channels_list:
            self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, 1))
            self.layer_blocks.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

        self._out_features = [str(i) for i in range(len(in_channels_list) + 2)]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                     nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        
        timm_features = self.timm_model(x) 
        
        # --- Транспонування (NHWC -> NCHW) ---
        corrected_features = []
        for feature, expected_channels in zip(timm_features, self.in_channels_list):
            if feature.dim() == 4 and feature.shape[-1] == expected_channels:
                 corrected_features.append(feature.permute(0, 3, 1, 2))
            else:
                 corrected_features.append(feature)
        
        timm_features = corrected_features
        # -------------------------------------
        
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
            
        p6, p7 = self.last_level(results[-1])
        results.append(p6)
        results.append(p7)

        out = {}
        for name, feature in zip(self._out_features, results):
            out[name] = feature
        
        return out

# --- Додаткові рівні P6 та P7 для FPN ---
class LastLevelP6P7(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        
    def forward(self, p5: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p6 = self.p6(p5)
        p7 = self.p7(F_torch.relu(p6))
        return p6, p7

def create_fpn_backbone(backbone_type: str, pretrained: bool = True, input_img_size=None):
    """
    Створює backbone з FPN, включаючи ResNet та Swin Transformer.
    Приймає input_img_size для коректної ініціалізації timm моделей.
    """
    if timm is None:
        if 'swin' in backbone_type:
             raise ImportError("Для використання Swin Transformer потрібна бібліотека 'timm'.")

    out_channels = 256
    timm_backbone_base = None
    
    # ------------------------------------------------------------------
    # --- ЛОГІКА ДЛЯ TIMM BACKBONES (Swin) ---
    # ------------------------------------------------------------------
    if 'swin' in backbone_type:
        
        timm_model_out_indices = (1, 2, 3) 
            
        if isinstance(input_img_size, int):
            size_for_timm = (input_img_size, input_img_size)
        elif isinstance(input_img_size, tuple) and len(input_img_size) == 2:
            size_for_timm = input_img_size
        else:
            size_for_timm = (800, 800) 
            
        timm_args = {'img_size': size_for_timm}

        timm_model_base = timm.create_model(
            backbone_type,
            features_only=True,
            out_indices=timm_model_out_indices, 
            pretrained=pretrained,
            **timm_args
        )
        
        in_channels_list = timm_model_base.feature_info.channels()
        
        backbone = CustomFPN(timm_model_base, in_channels_list, out_channels)
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