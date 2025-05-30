import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm # For progress bar
from CustomObjectDetectionDataset import CustomObjectDetectionDataset

# --- Configuration ---
# Set the path to your dataset
DATA_DIR = 'Data'

# Define the number of classes (e.g., if you have 'car' and 'person', num_classes = 2 + 1 for background)
# IMPORTANT: Adjust this based on your actual number of classes.
NUM_CLASSES = 1  # Example: 1 for 'object' + 1 for background. If you have 2 classes, set to 3.
# If you have specific class names, you can map them to integers starting from 1 (0 is background)
# Example: CLASS_NAMES = ['background', 'car', 'person']

# Training parameters
BATCH_SIZE = 2
LEARNING_RATE = 0.005
NUM_EPOCHS = 10
TRAIN_SPLIT_RATIO = 0.8 # 80% for training, 20% for validation

# Set device to GPU if available, otherwise CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

from CustomObjectDetectionDataset import CustomObjectDetectionDataset
# --- 2. Data Augmentations and Transforms ---
def get_transform(train):
    transforms = []
    transforms.append(torchvision.transforms.ToTensor())
    if train:
        # Add more transforms for data augmentation during training if needed
        # Example: transforms.append(T.RandomHorizontalFlip(0.5))
        pass
    return torchvision.transforms.Compose(transforms)

# --- Collate function for DataLoader ---
# This is necessary because the targets are dictionaries and not fixed-size tensors
def collate_fn(batch):
    return tuple(zip(*batch))

# --- 3. Model Definition ---
def get_faster_rcnn_model(num_classes):
    # Load a pre-trained Faster R-CNN model with a ResNet-50 FPN backbone
    # Use the default weights for pre-training on COCO dataset
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one that has the number of classes we desire
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# --- 4. Training Function ---
def train_model(model, data_loader, optimizer, num_epochs, device):
    model.train() # Set model to training mode
    for epoch in range(num_epochs):
        total_loss = 0
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        # Use tqdm for a progress bar
        for images, targets in tqdm(data_loader, desc=f"Training Epoch {epoch+1}"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        print(f"Epoch {epoch + 1} finished. Total Loss: {total_loss:.4f}")

# --- Main execution ---
if __name__ == "__main__":
    # Initialize dataset
    dataset = CustomObjectDetectionDataset(
        data_dir=DATA_DIR,
        transforms=get_transform(train=True)
    )

    # Split dataset into training and validation
    train_size = int(TRAIN_SPLIT_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0, # Set to 0 for Windows, or if you encounter multiprocessing issues
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0, # Set to 0 for Windows, or if you encounter multiprocessing issues
        collate_fn=collate_fn
    )

    # Get the model
    model = get_faster_rcnn_model(NUM_CLASSES + 1) # +1 for background class
    model.to(DEVICE)

    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    # Define learning rate scheduler (optional but recommended)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print("\nStarting training...")
    train_model(model, train_loader, optimizer, NUM_EPOCHS, DEVICE)
    print("Training complete!")

    # Save the trained model (optional)
    torch.save(model.state_dict(), 'faster_rcnn_model.pth')
    print("Model saved to faster_rcnn_model.pth")
