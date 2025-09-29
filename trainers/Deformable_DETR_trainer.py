# Deformable_DETR_trainer.py

from trainers.DETR_trainer import DETRTrainer
from transformers import DeformableDetrImageProcessor

class DeformableDETRTrainer(DETRTrainer):
    """Керує процесом навчання моделі Deformable DETR."""

    def _get_project_name(self):
        return self.params.get('project', 'runs/deformable-detr')

    def _get_model_checkpoint(self):
        return "SenseTime/deformable-detr"
    
    def _get_model_name(self):
        return "DeformableDETR"

    def _initialize_processor(self):
        # Deformable DETR використовує свій специфічний процесор
        return DeformableDetrImageProcessor.from_pretrained(self._get_model_checkpoint())