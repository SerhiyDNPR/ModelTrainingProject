# detr_wrapper.py

import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
from Validators.model_wrapper import ModelWrapper
from Validators.prediction import Prediction

class DETRWrapper(ModelWrapper):
    """Обгортка для моделей DETR."""

    def __init__(self, class_names, device):
        super().__init__(class_names, device)
        self.image_processor = None

    def load(self, model_path):
        try:
            num_classes = len(self.class_names)
            self.image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            
            model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            in_features = model.class_labels_classifier.in_features
            model.class_labels_classifier = torch.nn.Linear(in_features, num_classes + 1)
            
            id2label = {i: name for i, name in enumerate(self.class_names)}
            model.config.id2label = id2label
            model.config.label2id = {name: i for i, name in id2label.items()}

            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            self.model = model.to(self.device).eval()
            print(f"✅ Модель DETR '{model_path}' успішно завантажена.")
        except Exception as e:
            print(f"❌ Помилка завантаження моделі DETR: {e}")
            raise

    def predict(self, frame, conf_threshold):
        predictions = []
        
        # Конвертація BGR (OpenCV) -> RGB -> PIL
        rgb_frame = frame[:, :, ::-1]
        pil_image = Image.fromarray(rgb_frame)
        
        inputs = self.image_processor(images=pil_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]], device=self.device)
        results = self.image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=conf_threshold)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            class_id = label.item()
            class_name = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else "Unknown"
            
            predictions.append(
                Prediction(
                    box.cpu().numpy(),
                    score.item(),
                    class_id,
                    class_name
                )
            )
        return predictions