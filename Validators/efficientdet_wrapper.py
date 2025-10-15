# Validators/efficientdet_wrapper.py

import torch
from torchvision.transforms import functional as F
from Validators.model_wrapper import ModelWrapper
from Validators.prediction import Prediction
import numpy as np

# EfficientDet вимагає сторонньої бібліотеки effdet
try:
    from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
    from effdet.efficientdet import HeadNet
except ImportError:
    print("Помилка: бібліотеку 'effdet' не знайдено.")
    print("Будь ласка, встановіть її командою: pip install effdet")
    exit(1)

# Словник з конфігураціями моделей: назва та рекомендований розмір зображення
BACKBONE_CONFIGS = {
    '1': ('tf_efficientdet_d0', (512, 512)),
    '2': ('tf_efficientdet_d1', (640, 640)),
    '3': ('tf_efficientdet_d2', (768, 768)),
    '4': ('tf_efficientdet_d3', (896, 896)),
    '5': ('tf_efficientdet_d4', (1024, 1024)),
    '6': ('tf_efficientdet_d5', (1280, 1280)),
    '7': ('tf_efficientdet_d6', (1536, 1536)),
    '8': ('tf_efficientdet_d7', (1536, 1536)),
}

class EfficientDetWrapper(ModelWrapper):
    """Обгортка для моделей EfficientDet, навчених через EfficientDet_trainer.py."""

    def __init__(self, class_names, device):
        super().__init__(class_names, device)
        self.image_size = None # Буде зберігати обраний розмір зображення

    def _select_backbone(self):
        """Відображає меню вибору backbone і повертає вибір користувача."""
        print("\nБудь ласка, оберіть 'хребет' (backbone) для моделі EfficientDet, що завантажується:")
        for key, (name, size) in BACKBONE_CONFIGS.items():
            model_id = name.replace('tf_efficientdet_', '').upper()
            print(f"  {key}: {model_id:<4} (розмір: {size[0]}x{size[1]})")
        
        while True:
            choice = input(f"Ваш вибір (1-{len(BACKBONE_CONFIGS)}): ").strip()
            if choice in BACKBONE_CONFIGS:
                model_name, image_size = BACKBONE_CONFIGS[choice]
                print(f"✅ Обрано архітектуру: {model_name.upper()}")
                return model_name, image_size
            else:
                print(f"❌ Невірний вибір. Будь ласка, введіть число від 1 до {len(BACKBONE_CONFIGS)}.")

    def load(self, model_path):
        """Завантажує модель EfficientDet та адаптує її."""
        try:
            backbone_name, self.image_size = self._select_backbone()
            num_classes = len(self.class_names)

            # 1. Створюємо конфігурацію моделі
            config = get_efficientdet_config(backbone_name)
            config.num_classes = num_classes
            config.image_size = self.image_size
            
            # 2. Створюємо архітектуру, як у трейнері
            model = EfficientDet(config, pretrained_backbone=False) # Ваги завантажимо з файлу
            model.class_net = HeadNet(config, num_outputs=num_classes)
            
            # 3. Завантажуємо навчені ваги
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)

            # 4. Обгортаємо модель у DetBenchPredict для інференсу
            self.model = DetBenchPredict(model).to(self.device).eval()
            print(f"✅ Модель EfficientDet ({backbone_name.upper()}) '{model_path}' успішно завантажена.")
        
        except Exception as e:
            print(f"❌ Помилка завантаження моделі EfficientDet: {e}")
            raise

    def predict(self, frame, conf_threshold):
        """Робить передбачення на одному кадрі."""
        predictions = []
        
        # Конвертація BGR (OpenCV) -> RGB -> Tensor
        rgb_frame = frame[:, :, ::-1].copy()
        tensor_frame = F.to_tensor(rgb_frame).to(self.device)
        tensor_frame = tensor_frame.unsqueeze(0) # Додаємо batch dimension

        with torch.no_grad():
            results = self.model(tensor_frame)

        # Результати вже відфільтровані за score всередині DetBenchPredict
        # Формат: [batch_idx, x1, y1, x2, y2, score, class_id]
        for det in results[0]:
            score = det[4].item()
            if score >= conf_threshold:
                box = det[0:4].cpu().numpy()
                class_id = int(det[5].item())
                
                if 0 <= class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    predictions.append(
                        Prediction(
                            box,
                            score,
                            class_id,
                            class_name
                        )
                    )
        return predictions