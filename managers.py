# managers.py
"""
Містить клас-менеджер, що оркеструє весь робочий процес.
"""
import sys
import datetime as dt
from config import FRAMEWORKS
from converters.YOLO_converter import YOLODataConverter
from converters.ResNET_converter import ResNetDataConverter
from converters.PascalVOC_converter import PascalVOCDataConverter
from converters.COCO_converter import COCODataConverter
from trainers.YOLO_trainer import YOLOTrainer
from trainers.ResNET_trainer import ResNetTrainer
from trainers.FasterRCNNTrainer import FasterRCNNTrainer
from trainers.DETR_trainer import DETRTrainer
from trainers.Deformable_DETR_trainer import DeformableDETRTrainer
from trainers.FCOS_trainer import FCOSTrainer
from trainers.RT_DETR_Ultralytics_trainer import RTDETRUltralyticsTrainer
from trainers.RetinaNet_trainer import RetinaNetTrainer
from trainers.MaskRCNN_trainer import MaskRCNNTrainer
from trainers.CascadeRCNN_trainer import CascadeRCNNTrainer
from trainers.SSDTrainer import SSDTrainer
from trainers.EfficientDet_trainer import EfficientDetTrainer

class TrainingManager:
    """Керує повним циклом: вибір фреймворку, конвертація, навчання."""
    def __init__(self, config):
        self.config = config
        self.framework_name = None
        self.converter = None
        self.trainer = None

    def _select_framework(self):
        """Запитує у користувача, який фреймворк використовувати."""
        print("Будь ласка, оберіть фреймворк для навчання:")
        for key, value in FRAMEWORKS.items():
            print(f"  {key}: {value}")
        
        try:
            choice = int(input("Ваш вибір: "))
            if choice in FRAMEWORKS:
                self.framework_name = FRAMEWORKS[choice]
                print(f"Ви обрали: {self.framework_name}")
            else:
                print("Невірний вибір. Спробуйте ще раз.")
                sys.exit(1)
        except ValueError:
            print("Будь ласка, введіть число.")
            sys.exit(1)

    def _initialize_components(self):
        if "YOLO" in self.framework_name:
            self.converter = YOLODataConverter(
                source_dir=self.config.PERCEPTION_SOURCE_DIR,
                output_dir=self.config.FINAL_DATASET_DIR
            )
            self.trainer = YOLOTrainer(
                training_params=self.config.YOLO_TRAIN_PARAMS,
                dataset_dir=self.config.FINAL_DATASET_DIR
            )
        elif self.framework_name == "ResNet (clasification only - dead end)":
            self.converter = ResNetDataConverter(
                source_dir=self.config.PERCEPTION_SOURCE_DIR,
                output_dir="ResNetDataset"
            )
            self.trainer = ResNetTrainer(
                training_params=self.config.RESNET_TRAIN_PARAMS,
                dataset_dir="ResNetDataset"
            )
        elif self.framework_name.startswith("Faster R-CNN"):
            output_folder = "PascalVOCDataset_FasterRCNN"
            self.converter = PascalVOCDataConverter(
                source_dir=self.config.PERCEPTION_SOURCE_DIR,
                output_dir=output_folder
            )
            self.trainer = FasterRCNNTrainer(
                training_params=self.config.FASTER_RCNN_TRAIN_PARAMS,
                dataset_dir=output_folder
            )
        elif self.framework_name == "DETR":
            output_folder = "COCODataSet_DETR"
            self.converter = COCODataConverter(
                source_dir=self.config.PERCEPTION_SOURCE_DIR,
                output_dir=output_folder
            )
            self.trainer = DETRTrainer(
                training_params=self.config.DETR_TRAIN_PARAMS,
                dataset_dir=output_folder
            )
        elif self.framework_name == "Deformable DETR":
            output_folder = "COCODataSet_DeformableDETR"
            self.converter = COCODataConverter(
                source_dir=self.config.PERCEPTION_SOURCE_DIR,
                output_dir=output_folder
            )
            self.trainer = DeformableDETRTrainer(
                training_params=self.config.DEFORMABLE_DETR_TRAIN_PARAMS,
                dataset_dir=output_folder
            )
        elif self.framework_name == "FCOS":
            output_folder = "COCODataSet_FCOS"
            self.converter = COCODataConverter(
                source_dir=self.config.PERCEPTION_SOURCE_DIR,
                output_dir=output_folder
            )
            self.trainer = FCOSTrainer(
                training_params=self.config.FCOS_TRAIN_PARAMS,
                dataset_dir=output_folder
            )
        elif self.framework_name == "EfficientDet":
            # EfficientDet, як і FCOS, добре працює з COCO
            output_folder = "COCODataSet_EfficientDet"
            self.converter = COCODataConverter(
                source_dir=self.config.PERCEPTION_SOURCE_DIR,
                output_dir=output_folder
            )
            self.trainer = EfficientDetTrainer(
                training_params=self.config.EFFICIENTDET_TRAIN_PARAMS,
                dataset_dir=output_folder
            )            
        elif self.framework_name == "RT-DETR (Ultralytics)":
            self.converter = YOLODataConverter(
                source_dir=self.config.PERCEPTION_SOURCE_DIR,
                output_dir="YoloDataset_For_RTDETR"
            )
            self.trainer = RTDETRUltralyticsTrainer(
                training_params=self.config.RT_DETR_TRAIN_PARAMS,
                dataset_dir="YoloDataset_For_RTDETR"
            )
        elif self.framework_name == "Mask R-CNN":
            # Mask R-CNN може використовувати той же конвертер, що і Faster R-CNN
            output_folder = "PascalVOCDataSet_MaskRCNN"
            self.converter = PascalVOCDataConverter(
                source_dir=self.config.PERCEPTION_SOURCE_DIR,
                output_dir=output_folder
            )
            self.trainer = MaskRCNNTrainer(
                training_params=self.config.MASK_RCNN_TRAIN_PARAMS,
                dataset_dir=output_folder
            )
        elif self.framework_name.startswith("RetinaNet"):
            # RetinaNet, як і FCOS, добре працює з COCO
            output_folder = "COCODataSet_RetinaNet"
            self.converter = COCODataConverter(
                source_dir=self.config.PERCEPTION_SOURCE_DIR,
                output_dir=output_folder
            )
            self.trainer = RetinaNetTrainer(
                training_params=self.config.RETINANET_TRAIN_PARAMS,
                dataset_dir=output_folder
            )
        elif self.framework_name == "Cascade R-CNN":
            # Використовуємо той же конвертер, що і Faster R-CNN
            output_folder = "PascalVOCDataSet_CascadeRCNN"
            self.converter = PascalVOCDataConverter(
                source_dir=self.config.PERCEPTION_SOURCE_DIR,
                output_dir=output_folder
            )
            self.trainer = CascadeRCNNTrainer(
                training_params=self.config.CASCADE_RCNN_TRAIN_PARAMS,
                dataset_dir=output_folder
            )
        elif self.framework_name == "SSD":
            output_folder = "PascalVOCDataSet_SSD"
            self.converter = PascalVOCDataConverter(
                source_dir=self.config.PERCEPTION_SOURCE_DIR,
                output_dir=output_folder
            )
            self.trainer = SSDTrainer(
                training_params=self.config.SSD_TRAIN_PARAMS,
                dataset_dir=output_folder
            )
        else:
            raise ValueError(f"Обраний фреймворк '{self.framework_name}' не підтримується.")

    def _log_training_summary(self, summary_data):
        """Записує результати навчання у лог-файл."""
        log_file_path = "training_log.txt"
        
        hyperparams_str = "\n".join([f"    - {key}: {value}" for key, value in summary_data.get("hyperparameters", {}).items()])
        
        summary_str = f"""
==================================================
          Training Session Summary
==================================================
Model:              {summary_data.get("model_name", "N/A")}
Timestamp:          {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Dataset Info:
    - Image Count:      {summary_data.get("image_count", "N/A")}
    - Negative Count:   {summary_data.get("negative_count", "N/A")}
    - Class Count:      {summary_data.get("class_count", "N/A")}
    - Image Size:       {summary_data.get("image_size", "N/A")}

Training Results:
    - Best mAP50-95:    {summary_data.get("best_map", "N/A")}
    - Best Model Path:  {summary_data.get("best_model_path", "N/A")}

Hyperparameters:
{hyperparams_str}
--------------------------------------------------
"""
        try:
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(summary_str)
            print(f"\n✅ Результати навчання успішно записано у файл: {log_file_path}")
        except Exception as e:
            print(f"\n❌ Помилка при записі лог-файлу: {e}")

    def run(self):
        """Запускає основний робочий процес."""
        self._select_framework()
        self._initialize_components()

        dataset_stats = {}
        do_conversion = input("\nБажаєте запустити конвертацію даних? (y/n): ").strip().lower()
        if do_conversion in ['y', 'Y', 'н', 'Н']:
            # Конвертери тепер мають повертати словник зі статистикою
            dataset_stats = self.converter.prepare_data()
        else:
            print("Конвертацію пропущено. Спроба визначити розмір зображення з вихідних даних...")
            if hasattr(self.converter, 'get_image_dimensions'):
                imgsize = self.converter.get_image_dimensions()
                dataset_stats['image_size'] = imgsize
            else:
                dataset_stats['image_size'] = None

        if not dataset_stats or not dataset_stats.get('image_size'):
            print("\nПОМИЛКА: Не вдалося визначити розмір зображення. Навчання неможливе.")
            sys.exit(1)
        
        print(f"\n✅ Розмір зображення для навчання встановлено: {dataset_stats['image_size']}")

        # Виводимо гіперпараметри перед навчанням
        self.trainer.display_hyperparameters()

        # Запускаємо навчання і отримуємо результати
        results_summary = self.trainer.start_or_resume_training(dataset_stats)
        
        # Записуємо результати у лог-файл
        if results_summary:
            self._log_training_summary(results_summary)