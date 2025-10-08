import torch

# Шлях до папки Unity Perception
PERCEPTION_SOURCE_DIR = r"C:\Users\serhi\AppData\LocalLow\DefaultCompany\GenerateSynteticData"

# Назва папки, куди будуть збережені конвертовані дані для навчання
FINAL_DATASET_DIR = "YoloDataset"


# --- Параметри фреймворків ---
FRAMEWORKS = {
    1: "YOLO",
    2: "ResNet (clasification only - dead end)",
    3: "Faster R-CNN (ResNet50/ResNet101/MobileNet)", 
    4: "DETR",
    5: "Deformable DETR",    
    6: "FCOS",
    7: "RT-DETR (Ultralytics)",
    8: "Mask R-CNN",
    9: "RetinaNet",
    10: "Cascade R-CNN (NOT IMPLEMENTED)",
    11: "SSD"
}

# Параметри для навчання YOLO
YOLO_TRAIN_PARAMS = {
    'epochs': 40,
    'batch': -1,
    'amp': True, 
    'patience': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'project': 'runs/detect', # Головна папка для збереження результатів
}

# Параметри для навчання ResNet
RESNET_TRAIN_PARAMS = {
    'epochs': 25,
    'batch_size': 32,
    'learning_rate': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# Параметри для навчання Faster R-CNN
FASTER_RCNN_TRAIN_PARAMS = {
    'epochs': 25,
    'batch': 32, # Зменшено для економії пам'яті на Acer 4х4 ResNet50, 8x2 Mobile net, 
                 # на домашньому MobileNet можна гнати на 32 при 1068х800
    'accumulation_steps': 1, 
    'lr': 0.0001, # Типове значення для AdamW
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'project': 'runs/faster-rcnn',
    'lr_scheduler_step_size': 8, # Зменшувати LR кожні 8 епох
    'lr_scheduler_gamma': 0.1,   # Зменшувати LR в 10 разів (0.1)
}

# Параметри для навчання DETR (взято з RCNN як схожі... потребує перевірки)
DETR_TRAIN_PARAMS = {
    'epochs': 25,
    'batch': 8,
    'accumulation_steps': 4,  # Встановлено кроки накопичення (batch * accumulation_steps = 16)    
    'lr': 1e-4, # Основний LR для Transformer
    'lr_backbone': 1e-5, # LR для Backbone
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'project': 'runs/DETR',
    # Додано параметри для планувальника LR
    'lr_scheduler_step_size': 15, # Зменшувати LR кожні 15 епох
    'lr_scheduler_gamma': 0.1,   # Зменшувати LR в 10 разів (0.1)
}

# Параметри для навчання Deformable DETR
DEFORMABLE_DETR_TRAIN_PARAMS = {
    'epochs': 25,
    'batch': 4,  
    'accumulation_steps': 4,  # Встановлено кроки накопичення (batch * accumulation_steps = 16)
    'lr': 2e-4,
    'lr_backbone': 2e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'project': 'runs/DeformableDETR',
    'lr_scheduler_step_size': 15,
    'lr_scheduler_gamma': 0.1,
}

# Параметри для навчання FCOS
FCOS_TRAIN_PARAMS = {
    'epochs': 25,
    'batch': 8,
    'lr': 0.0001, # Стандартний 'lr' для оптимізатора AdamW (1e-4)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'project': 'runs/fcos',
    # Параметри планувальника можна залишити, AdamW також добре з ним працює
    'lr_scheduler_step_size': 8,
    'lr_scheduler_gamma': 0.1,
}

# Параметри для навчання RT-DETR через Ultralytics (схожі на YOLO)
RT_DETR_TRAIN_PARAMS = {
    'epochs': 50,
    'batch': -1, # optimizer=auto по замовчанню, хай підбирає, він справляється краще (але не вгадує максимум памяті)
    'optimizer': 'AdamW', # оптимізатор
    'lr0': 0.0001,        # менша швидкість навчання на перших парах був "випбух градієнтів"
    'patience': 15,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'project': 'runs/rtdetr_ultralytics',
}

# Параметри для навчання Mask R-CNN
MASK_RCNN_TRAIN_PARAMS = {
    'epochs': 25,
    'batch': 2,
    'accumulation_steps': 8, # Дуже вимоглива до пам'яті
    'lr': 0.0001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'project': 'runs/mask-rcnn', 
    'lr_scheduler_step_size': 8,
    'lr_scheduler_gamma': 0.1,
}

# Параметри для навчання RetinaNet
RETINANET_TRAIN_PARAMS = {
    'epochs': 25,
    'batch': 2,
    'accumulation_steps': 8,
    'lr': 0.0001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'project': 'runs/retinanet',
    'lr_scheduler_step_size': 8,
    'lr_scheduler_gamma': 0.1,
}

# Параметри для Cascade R-CNN 
CASCADE_RCNN_TRAIN_PARAMS = {
    'epochs': 25,
    'batch': 4,  # Зменшено, оскільки модель вимогливіша до пам'яті
    'accumulation_steps': 4,  # Збільшено для компенсації меншого batch_size (4x4=16)
    'lr': 0.0001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'project': 'runs/cascade-rcnn',
    'lr_scheduler_step_size': 8,
    'lr_scheduler_gamma': 0.1,
}

SSD_TRAIN_PARAMS = {
    'epochs': 30,
    'batch': 16, # SSD досить вимоглива до пам'яті
    'accumulation_steps': 2, # 4x4=16 ефективний batch_size
    'lr': 0.00001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'project': 'runs/ssd-vgg16',
    'lr_scheduler_step_size': 8, # Зменшувати LR кожні 10 епох
    'lr_scheduler_gamma': 0.1,    # Зменшувати LR в 10 разів
}
