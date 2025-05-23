import os
from glob import glob
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# === Завантаження моделі ===
model = YOLO("square_detector.pt")

# === Шляхи до тестової вибірки ===
test_images_dir = "YoloDataset/images/test"
test_labels_dir = "YoloDataset/labels/test"

image_paths = sorted(glob(os.path.join(test_images_dir, "*.jpg")))

ious = []
correct_detections = 0
missed_detections = 0
false_positives = 0

for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")
    result = model.predict(image, imgsz=640, conf=0.5)[0]

    label_path = os.path.join(test_labels_dir, os.path.basename(image_path).replace(".jpg", ".txt"))
    with open(label_path, "r") as f:
        label_data = f.read().strip()

    gt_has_object = bool(label_data)
    pred_boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) > 0 else []

    if gt_has_object:
        gt_line = label_data.split()[1:]  # skip class id
        x_center, y_center, w, h = map(float, gt_line)
        img_w, img_h = image.size
        x1 = (x_center - w / 2) * img_w
        y1 = (y_center - h / 2) * img_h
        x2 = (x_center + w / 2) * img_w
        y2 = (y_center + h / 2) * img_h
        gt_box = [x1, y1, x2, y2]

        if len(pred_boxes) > 0:
            best_iou = max(compute_iou(gt_box, pred_box) for pred_box in pred_boxes)
            ious.append(best_iou)
            if best_iou >= 0.5:
                correct_detections += 1
            else:
                missed_detections += 1
        else:
            missed_detections += 1
    else:
        if len(pred_boxes) > 0:
            false_positives += 1

# === Результати ===
total = len(image_paths)
print(f"Total images: {total}")
print(f"Correct detections: {correct_detections}")
print(f"Missed detections: {missed_detections}")
print(f"False positives: {false_positives}")

precision = correct_detections / (correct_detections + false_positives) if (correct_detections + false_positives) else 0
recall = correct_detections / (correct_detections + missed_detections) if (correct_detections + missed_detections) else 0
mean_iou = np.mean(ious) if ious else 0

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"Mean IoU: {mean_iou:.3f}")

# Псевдо mAP@0.5 для 1-класової задачі з 1 GT bbox на зображення:
print(f"Approximate mAP@0.5: {recall * precision:.3f}")
