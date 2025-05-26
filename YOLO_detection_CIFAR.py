import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from ultralytics import YOLO
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm
from Image_generator.ImageGeneratorLib import draw_random_ufo 

# === Завантаження публічної бібліотеки зображень ===
cifar = CIFAR10(root='.', download=True)
images = [Image.fromarray(np.array(cifar[i][0])) for i in range(len(cifar))]

# === Завантаження навченого детектора YOLO ===
model = YOLO("UFO_detector.pt")

# === Додавання випадкового квадрата до зображення ===
def generate_random_color():
    return tuple([random.randint(0, 255) for _ in range(3)])

def add_random_square(image):
    image = image.convert("RGB").resize((640, 480))
    draw = ImageDraw.Draw(image)
    width, height = image.size

    center_x = random.randint(100, 540)
    center_y = random.randint(100, 380)
    size = random.randint(30, 100)
    angle = random.uniform(0, 360)
    scale = random.uniform(0.5, 1.5)
    half_size = size * scale / 2

    pts = np.array([
        [-half_size, -half_size],
        [ half_size, -half_size],
        [ half_size,  half_size],
        [-half_size,  half_size]
    ])

    theta = math.radians(angle)
    c, s = math.cos(theta), math.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]])
    rotated_pts = np.dot(pts, rotation_matrix) + [center_x, center_y]

    polygon = [tuple(p) for p in rotated_pts]
    color = generate_random_color()
    draw.polygon(polygon, fill=color)

    xs = rotated_pts[:, 0]
    ys = rotated_pts[:, 1]
    x_min, x_max = max(0, min(xs)), min(width, max(xs))
    y_min, y_max = max(0, min(ys)), min(height, max(ys))
    return image.filter(ImageFilter.GaussianBlur(radius=1)), [x_min, y_min, x_max, y_max]

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

# === Прогон по всіх зображеннях з обрахунком метрик ===
ious = []
TP = FP = FN = TN = 0

for idx, img in enumerate(tqdm(images[:100], desc="Evaluating on synthetic data")):
    has_square = idx % 2 == 0
    if has_square:
        modified, gt_box = add_random_square(img)
    else:
        modified = img.convert("RGB").resize((640, 480)).filter(ImageFilter.GaussianBlur(radius=1))
        gt_box = None

    result = model.predict(modified, imgsz=640, conf=0.5)[0]
    pred_boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else []

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
