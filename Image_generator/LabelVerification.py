import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# === Параметри ===
images_dir = "Data"
image_prefix = "Image"
image_ext = ".jpg"
label_ext = ".txt"

# === Завантаження зображення та розмітки ===
def load_and_display(image_id):
    image_path = os.path.join(images_dir, f"{image_prefix}{image_id:04d}{image_ext}")
    label_path = os.path.join(images_dir, f"{image_prefix}{image_id:04d}{label_ext}")

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, x_center, y_center, w, h = map(float, parts)
                x0 = (x_center - w / 2) * width
                y0 = (y_center - h / 2) * height
                x1 = (x_center + w / 2) * width
                y1 = (y_center + h / 2) * height
                draw.rectangle([x0, y0, x1, y1], outline="red", width=2)

    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Image {image_id:03d}")
    plt.show()

# === Приклад використання ===
for i in range(10):
    load_and_display(i+1)
