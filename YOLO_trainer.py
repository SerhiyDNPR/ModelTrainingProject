# YOLO training setup using Ultralytics YOLOv5 (PyTorch-based)
# Ensure the 'ultralytics' package is installed via: pip install ultralytics

import os
import random
import shutil
from glob import glob
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

def prepare_yolo_folder(yolo_folder="YoloDataset"):
    os.makedirs(yolo_folder, exist_ok=True)
    for filename in os.listdir(yolo_folder):
        file_path = os.path.join(yolo_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Step 1: Prepare dataset directories
data_dir = "Data"
dataset_dir = "YoloDataset"

prepare_yolo_folder(dataset_dir)

images_dir = os.path.join(dataset_dir, "images")
labels_dir = os.path.join(dataset_dir, "labels")

for subset in ["train", "val", "test"]:
    os.makedirs(os.path.join(images_dir, subset), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, subset), exist_ok=True)

# Step 2: Collect all images and their corresponding labels
image_paths = sorted(glob(os.path.join(data_dir, "*.jpg")))
label_paths = [path.replace(".jpg", ".txt") for path in image_paths]

# Step 3: Split data
train_imgs, test_imgs, train_lbls, test_lbls = train_test_split(image_paths, label_paths, test_size=0.2, random_state=42)
val_imgs, test_imgs, val_lbls, test_lbls = train_test_split(test_imgs, test_lbls, test_size=0.5, random_state=42)

splits = {
    "train": (train_imgs, train_lbls),
    "val": (val_imgs, val_lbls),
    "test": (test_imgs, test_lbls)
}

# Step 4: Move files to appropriate folders
for split, (imgs, lbls) in splits.items():
    for img, lbl in zip(imgs, lbls):
        shutil.copy(img, os.path.join(images_dir, split, os.path.basename(img)))
        shutil.copy(lbl, os.path.join(labels_dir, split, os.path.basename(lbl)))

# Step 5: Create dataset config
with open("yolo_config.yaml", "w") as f:
    f.write("""
path: YoloDataset
train: images/train
val: images/val
test: images/test

nc: 1
names: ['UFO']
""")

# Step 6: Train the model
model = YOLO("yolov8n.pt")  # Or use "yolov8n.pt" for pretrained weights
model.train(data="yolo_config.yaml", epochs=25, imgsz=640)

# Train the model
model.train(
    data='yolo_config.yaml',       # dataset config
    epochs=25,
    imgsz=640,
    batch=16,
    project='runs/train',           # default path for logs
    name='UFO_training_tensorboard',         # experiment name
    exist_ok=True
)

# Step 7: Save the model
model.save("square_detector.pt")
