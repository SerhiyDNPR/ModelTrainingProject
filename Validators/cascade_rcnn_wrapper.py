# Validators/cascade_rcnn_wrapper.py

import sys
import torch
from torchvision.transforms import functional as F

from .model_wrapper import ModelWrapper
from .prediction import Prediction

# Імпорти, специфічні для MMDetection, як у тренері
try:
    from mmengine.config import Config
    from mmengine.registry import MODELS
    import mmengine
    from mmdet.structures import DetDataSample
    
    # Явний імпорт для реєстрації компонентів
    from mmdet.models.detectors import *
    from mmdet.models.backbones import *
    from mmdet.models.necks import *
    from mmdet.models.roi_heads import *
    from mmdet.models.dense_heads import *
except ImportError:
    print("="*60)
    print("🔴 ПОМИЛКА: MMDetection або його залежності не встановлено!")
    print("   Будь ласка, встановіть їх згідно з інструкцією до CascadeRCNN_trainer,")
    print("   інакше валідація Cascade R-CNN буде неможливою.")
    print("="*60)
    sys.exit(1)


class CascadeRCNNWrapper(ModelWrapper):
    """Обгортка для валідації моделей Cascade R-CNN, навчених через MMDetection."""

    def _select_backbone(self):
        """
        Відображає меню вибору backbone, щоб правильно відтворити архітектуру моделі.
        Це необхідно, оскільки файл ваг (.pth) не містить інформації про архітектуру.
        """
        print("\nБудь ласка, оберіть 'хребет' (backbone), з яким було навчено модель Cascade R-CNN:")
        print("  1: ResNet-50")
        print("  2: ResNet-101")
        
        while True:
            choice = input("Ваш вибір (1 або 2): ").strip()
            if choice == '1':
                print("✅ Обрано архітектуру на базі ResNet-50.")
                return 'resnet50'
            elif choice == '2':
                print("✅ Обрано архітектуру на базі ResNet-101.")
                return 'resnet101'
            else:
                print("❌ Невірний вибір. Будь ласка, введіть 1 або 2.")

    def load(self, model_path):
        """Завантажує модель Cascade R-CNN, створюючи архітектуру та завантажуючи ваги."""
        try:
            backbone_type = self._select_backbone()

            if backbone_type == 'resnet50':
                config_path = 'configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py'
            elif backbone_type == 'resnet101':
                config_path = 'configs/cascade_rcnn/cascade-rcnn_r101_fpn_1x_coco.py'
            else:
                raise ValueError(f"Непідтримуваний backbone '{backbone_type}'.")

            cfg = Config.fromfile(config_path)
            num_classes = len(self.class_names)
            cfg.model.roi_head.bbox_head[0].num_classes = num_classes
            cfg.model.roi_head.bbox_head[1].num_classes = num_classes
            cfg.model.roi_head.bbox_head[2].num_classes = num_classes
            
            mmengine.DefaultScope.get_instance('mmdet_scope', scope_name='mmdet')
            
            self.model = MODELS.build(cfg.model)
            checkpoint = torch.load(model_path, map_location='cpu')            
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            corrected_state_dict = {key.replace("model.", "", 1): value for key, value in state_dict.items()}
            
            self.model.load_state_dict(corrected_state_dict)
            
            # Цей рядок тепер коректно перемістить всю модель з CPU на цільовий пристрій.
            self.model = self.model.to(self.device).eval()
            
            print(f"✅ Модель Cascade R-CNN ({backbone_type.upper()}) '{model_path}' успішно завантажена.")
        
        except FileNotFoundError:
             print(f"❌ Помилка: конфігураційний файл '{config_path}' не знайдено.")
             print("   Переконайтесь, що директорія 'configs' з репозиторію MMDetection є у проекті.")
             sys.exit(1)
        except Exception as e:
            print(f"❌ Помилка завантаження моделі Cascade R-CNN: {e}")
            raise

    def predict(self, frame, conf_threshold):
        """Виконує детекцію на одному кадрі, використовуючи завантажену модель."""
        predictions = []
        
        rgb_frame = frame[:, :, ::-1].copy()
        tensor_frame = F.to_tensor(rgb_frame).to(self.device)
        
        with torch.no_grad():
            batched_tensor = tensor_frame.unsqueeze(0)

            data_sample = DetDataSample()
            
            data_sample.set_metainfo({
                'img_shape': tensor_frame.shape[1:],
                'ori_shape': frame.shape[:2],
                'scale_factor': (1.0, 1.0)
            })

            results_list = self.model.predict(batched_tensor, batch_data_samples=[data_sample])
            result = results_list[0]

        pred_instances = result.pred_instances
        
        for box, label, score in zip(pred_instances.bboxes, pred_instances.labels, pred_instances.scores):
            if score.item() >= conf_threshold:
                class_id = label.item()
                if 0 <= class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    predictions.append(
                        Prediction(
                            box.cpu().numpy(),
                            score.item(),
                            class_id,
                            class_name,
                            None  # track_id
                        )
                    )
        return predictions