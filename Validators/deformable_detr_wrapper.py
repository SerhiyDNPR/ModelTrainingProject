import torch
from transformers import DeformableDetrImageProcessor, DeformableDetrForObjectDetection
from PIL import Image
from Validators.model_wrapper import ModelWrapper
from Validators.prediction import Prediction

class DeformableDETRWrapper(ModelWrapper):
    """Обгортка для моделей Deformable DETR."""

    def __init__(self, class_names, device):
        super().__init__(class_names, device)
        self.image_processor = None

    def load(self, model_path):
        """
        Завантажує модель Deformable DETR та її процесор.
        """
        try:
            num_labels = len(self.class_names)

            # Завантажуємо процесор, що відповідає моделі Deformable DETR
            self.image_processor = DeformableDetrImageProcessor.from_pretrained("SenseTime/deformable-detr")

            # Створюємо мапу міток для конфігурації моделі
            id2label = {i: name for i, name in enumerate(self.class_names)}
            label2id = {name: i for i, name in id2label.items()}

            # Завантажуємо архітектуру моделі з потрібною кількістю класів
            # ignore_mismatched_sizes=True дозволяє завантажувати ваги в адаптовану "голову"
            model = DeformableDetrForObjectDetection.from_pretrained(
                "SenseTime/deformable-detr",
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True
            )

            # Завантажуємо навчені ваги з файлу чекпоінта
            checkpoint = torch.load(model_path, map_location=self.device)
            # Припускаємо, що ваги збережені в ключі 'model_state_dict'
            model.load_state_dict(checkpoint['model_state_dict'])

            self.model = model.to(self.device).eval()
            print(f"✅ Модель Deformable DETR '{model_path}' успішно завантажена.")

        except Exception as e:
            print(f"❌ Помилка завантаження моделі Deformable DETR: {e}")
            raise

    def predict(self, frame, conf_threshold):
        """
        Робить передбачення на одному кадрі.
        """
        predictions = []

        # Конвертація BGR (OpenCV) -> RGB -> PIL
        rgb_frame = frame[:, :, ::-1]
        pil_image = Image.fromarray(rgb_frame)

        # Підготовка зображення за допомогою процесора
        inputs = self.image_processor(images=pil_image, return_tensors="pt").to(self.device)

        # Виконання передбачення
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Постобробка результатів для отримання рамок, оцінок та міток
        target_sizes = torch.tensor([pil_image.size[::-1]], device=self.device)
        results = self.image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=conf_threshold)[0]

        # Форматування результатів у вигляд списку об'єктів Prediction
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
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