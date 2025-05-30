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

# --- Configuration ---
# Set the path to your dataset
DATA_DIR = 'Data'
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
LABELS_DIR = os.path.join(DATA_DIR, 'labels')

# Define the number of classes (e.g., if you have 'car' and 'person', num_classes = 2 + 1 for background)
# IMPORTANT: Adjust this based on your actual number of classes.
NUM_CLASSES = 2  # Example: 1 for 'object' + 1 for background. If you have 2 classes, set to 3.
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

# --- 1. Custom Dataset Class ---
class CustomObjectDetectionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.image_files.sort() # Ensure consistent order

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_name)

        # Load image
        img = Image.open(img_path).convert("RGB")
        img_width, img_height = img.size

        boxes = []
        labels = []

        # Load labels (YOLO format: class_id x_center y_center width height)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split(' ')))
                    class_id = int(parts[0])
                    x_center, y_center, width, height = parts[1:]

                    # Convert YOLO format to [x_min, y_min, x_max, y_max]
                    # Faster R-CNN expects absolute pixel coordinates
                    x_min = (x_center - width / 2) * img_width
                    y_min = (y_center - height / 2) * img_height
                    x_max = (x_center + width / 2) * img_width
                    y_max = (y_center + height / 2) * img_height

                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id + 1) # Add 1 because Faster R-CNN expects class 0 as background

        # Convert to PyTorch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Create target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["image_id"] = torch.tensor([idx]) # Optional, but good practice for COCO evaluation
        # target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # Optional
        # target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64) # Optional

        if self.transforms:
            img = self.transforms(img)

        return img, target

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

# --- 5. Evaluation (Simplified) and Visualization ---
def visualize_predictions(model, dataset, num_images_to_show=3, score_threshold=0.5):
    model.eval() # Set model to evaluation mode
    plt.figure(figsize=(15, 5 * num_images_to_show))

    # Get a few random images from the dataset for visualization
    indices = np.random.choice(len(dataset), num_images_to_show, replace=False)

    for i, idx in enumerate(indices):
        img_tensor, target = dataset[idx]
        original_img = Image.fromarray(img_tensor.mul(255).permute(1, 2, 0).byte().numpy())

        with torch.no_grad():
            prediction = model([img_tensor.to(DEVICE)])

        ax = plt.subplot(num_images_to_show, 1, i + 1)
        ax.imshow(original_img)
        ax.set_title(f"Image {idx} - Predictions (Score > {score_threshold})")
        ax.axis('off')

        # Draw ground truth boxes (optional, for comparison)
        # for box, label in zip(target['boxes'], target['labels']):
        #     rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
        #                              linewidth=1, edgecolor='g', facecolor='none', linestyle='--')
        #     ax.add_patch(rect)
        #     ax.text(box[0], box[1]-5, f"GT: {label.item()}", color='g', fontsize=8)


        # Draw predicted boxes
        for element in range(len(prediction[0]['boxes'])):
            score = prediction[0]['scores'][element].item()
            if score > score_threshold:
                box = prediction[0]['boxes'][element].cpu().numpy()
                label = prediction[0]['labels'][element].item()
                rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                         linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(box[0], box[1]-5, f"Class: {label} (Score: {score:.2f})", color='r', fontsize=8)

    plt.tight_layout()
    plt.show()

# --- Main execution ---
if __name__ == "__main__":
    # Create dummy data directories and files if they don't exist for demonstration
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)
    if not os.path.exists(LABELS_DIR):
        os.makedirs(LABELS_DIR)

    # Create some dummy images and labels for testing
    if not os.listdir(IMAGES_DIR) or not os.listdir(LABELS_DIR):
        print("Creating dummy dataset for demonstration...")
        dummy_image_size = (640, 480)
        for i in range(10): # Create 10 dummy images
            dummy_img = Image.new('RGB', dummy_image_size, color = (i*20 % 255, i*30 % 255, i*40 % 255))
            dummy_img.save(os.path.join(IMAGES_DIR, f'dummy_image_{i:02d}.jpg'))

            # Create dummy YOLO labels: class_id x_center y_center width height
            # Example: one object of class 0 (which will be class 1 in Faster R-CNN)
            # and one object of class 1 (which will be class 2 in Faster R-CNN)
            with open(os.path.join(LABELS_DIR, f'dummy_image_{i:02d}.txt'), 'w') as f:
                # Object 1 (class 0 in YOLO, class 1 in Faster R-CNN)
                f.write(f"0 0.5 0.5 0.2 0.3\n")
                # Object 2 (class 1 in YOLO, class 2 in Faster R-CNN)
                f.write(f"1 0.2 0.8 0.1 0.15\n")
        print("Dummy dataset created.")

    # Initialize dataset
    dataset = CustomObjectDetectionDataset(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
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
    # torch.save(model.state_dict(), 'faster_rcnn_model.pth')
    # print("Model saved to faster_rcnn_model.pth")

    print("\nVisualizing predictions on validation set...")
    # Use the validation dataset for visualization, but ensure it uses the same transforms
    # without augmentation if you want to see the raw images.
    # For this example, I'll re-initialize a dataset for visualization without train transforms.
    # A better approach would be to have separate transform pipelines for train/val/inference.
    val_dataset_for_viz = CustomObjectDetectionDataset(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        transforms=get_transform(train=False) # No augmentation for visualization
    )
    visualize_predictions(model, val_dataset_for_viz, num_images_to_show=3)

