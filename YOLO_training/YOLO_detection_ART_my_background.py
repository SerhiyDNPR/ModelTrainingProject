import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from ultralytics import YOLO
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm
from Image_generator.ImageGeneratorLib import draw_random_ufo, yolo_label_to_box, draw_gradient_background, draw_detailed_background 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

width, height = 640, 480

# === Завантаження публічної бібліотеки зображень ===
cifar = CIFAR10(root='.', download=True)
images = [Image.fromarray(np.array(cifar[i][0])) for i in range(len(cifar))]

# === Завантаження навченого детектора YOLO ===
model = YOLO("UFO_detector.pt")

# === Додавання випадкового квадрата до зображення ===
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

def draw_all_detection_boxes(image_with_object, result):
    if result.boxes is not None and len(result.boxes) > 0:
        _, ax = plt.subplots(1)
        ax.imshow(image_with_object)
        for box, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                box[0],
                box[1] - 5,
                f"{score:.2f}",
                color='yellow',
                fontsize=10,
                bbox=dict(facecolor='black', alpha=0.5, pad=0)
            )
        plt.axis('off')
        plt.show()

# === Прогон по всіх зображеннях з обрахунком метрик ===
ious = []
TP = FP = FN = TN = 0

for idx, img in enumerate(tqdm(images[:100], desc="Evaluating on synthetic data")):
    has_object = idx % 2 == 0

    #image_with_object = img.convert("RGB").resize((width, height)).filter(ImageFilter.GaussianBlur(radius=1))

    image_with_object = draw_gradient_background(width, height)

    draw = ImageDraw.Draw(image_with_object)

    yolo_label = ""
    draw_detailed_background(draw, width, height)

    gt_box = None

    if has_object:
        image_with_object, gt_label = draw_random_ufo(image_with_object)
        gt_box = yolo_label_to_box(gt_label, image_with_object.width, image_with_object.height)

    result = model.predict(image_with_object, imgsz=640, conf=0.5)[0]
    pred_boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else []

    draw_all_detection_boxes(image_with_object, result)
    
    if gt_box:
        if len(pred_boxes) > 0:
            best_iou = max(compute_iou(gt_box, pred_box) for pred_box in pred_boxes)
            ious.append(best_iou)
            if best_iou >= 0.5:
                TP += 1
            else:
                FN += 1
        else:
            FN += 1
    else:
        if len(pred_boxes) > 0:
            FP += 1
        else:
            TN += 1

# === Результати ===
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
mean_iou = np.mean(ious) if ious else 0
map_50 = precision * recall

print(f"\n=== Evaluation Results ===")
print(f"TP: {TP}")
print(f"FP: {FP}")
print(f"FN: {FN}")
print(f"TN: {TN}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"Mean IoU: {mean_iou:.3f}")
print(f"Approximate mAP@0.5: {map_50:.3f}")
