import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# === Завантаження публічної бібліотеки зображень ===
cifar = CIFAR10(root='.', download=True)
images = [Image.fromarray(np.array(cifar[i][0])) for i in range(len(cifar))]

# === Завантаження навченого детектора YOLO ===
model = YOLO("square_detector.pt")

# === Додавання випадкового квадрата до зображення ===
def add_random_square(image):
    image = image.convert("RGB").resize((640, 480))
    draw = ImageDraw.Draw(image)

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
    draw.polygon(polygon, fill=(255, 0, 0))
    return image.filter(ImageFilter.GaussianBlur(radius=1))

# === Детектування і візуалізація ===
def detect_and_display(image):
    results = model.predict(image, imgsz=640, conf=0.5)
    res_plotted = results[0].plot()
    plt.imshow(res_plotted)
    plt.axis('off')
    plt.show()

# === Основний цикл ===
for img in images:
    modified = add_random_square(img)
    detect_and_display(modified)
    input("Натисніть Enter щоб перейти до наступного зображення...")
