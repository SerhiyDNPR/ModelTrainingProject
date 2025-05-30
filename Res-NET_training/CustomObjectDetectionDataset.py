import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class CustomObjectDetectionDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        """
        Args:
            data_dir (str): Directory containing both images and label files.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transforms = transforms

        # List all image files (jpg, jpeg, png)
        self.image_files = [
            f for f in os.listdir(data_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.image_files.sort()  # Ensure consistent order

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        while True:
            img_name = self.image_files[idx]
            img_path = os.path.join(self.data_dir, img_name)
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(self.data_dir, label_name)

            img = Image.open(img_path).convert("RGB")
            img_width, img_height = img.size

            boxes = []
            labels = []

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = list(map(float, line.strip().split(' ')))
                        class_id = int(parts[0])
                        x_center, y_center, width, height = parts[1:]
                        x_min = (x_center - width / 2) * img_width
                        y_min = (y_center - height / 2) * img_height
                        x_max = (x_center + width / 2) * img_width
                        y_max = (y_center + height / 2) * img_height
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(class_id + 1)

            # Skip images with no boxes
            if len(boxes) == 0:
                # Pick another random index
                idx = np.random.randint(0, len(self.image_files))
                continue

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels

            if self.transforms:
                img = self.transforms(img)

            return img, target