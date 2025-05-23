import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
import math
import os
from multiprocessing import Pool, cpu_count

def generate_random_color():
    return tuple([random.randint(0, 255) for _ in range(3)])

def draw_random_square(draw, image_size, width, height):
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

    # Calculate YOLO annotation (axis-aligned bounding box around rotated square)
    xs = rotated_pts[:, 0]
    ys = rotated_pts[:, 1]
    x_min, x_max = max(0, min(xs)), min(width, max(xs))
    y_min, y_max = max(0, min(ys)), min(height, max(ys))

    # YOLO format: class_id, x_center_rel, y_center_rel, width_rel, height_rel
    x_center = (x_min + x_max) / 2.0 / width
    y_center = (y_min + y_max) / 2.0 / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height

    return f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"

def draw_detailed_background(draw, width, height):
    num_elements = 2000
    for _ in range(num_elements):
        shape_type = random.choice(['circle', 'square', 'triangle'])
        color = generate_random_color()
        x = random.randint(0, width)
        y = random.randint(0, height)
        size = random.randint(2, 8)

        if shape_type == 'circle':
            draw.ellipse([
                (x - size//2, y - size//2),
                (x + size//2, y + size//2)
            ], fill=color)
        elif shape_type == 'square':
            draw.rectangle([
                (x - size//2, y - size//2),
                (x + size//2, y + size//2)
            ], fill=color)
        else:  # triangle
            triangle = [
                (x, y - size),
                (x - size, y + size),
                (x + size, y + size)
            ]
            draw.polygon(triangle, fill=color)                

def draw_gradient_background(width, height):
    color1 = np.array(generate_random_color(), dtype=np.float32)
    color2 = np.array(generate_random_color(), dtype=np.float32)
    direction = np.random.randn(2)
    direction = direction / np.linalg.norm(direction)

    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            t = (x * direction[0] + y * direction[1]) / (width * abs(direction[0]) + height * abs(direction[1]))
            t = max(0, min(1, t))
            color = (1 - t) * color1 + t * color2
            gradient[y, x] = color.astype(np.uint8)

    return Image.fromarray(gradient)

def generate_image(index):
    width, height = 640, 480
    image = draw_gradient_background(width, height)
    draw = ImageDraw.Draw(image)

    yolo_label = ""
    draw_detailed_background(draw, width, height)
    if random.random() < 0.5:
        yolo_label = draw_random_square(draw, (width, height), width, height)

    image = image.filter(ImageFilter.GaussianBlur(radius=1))

    os.makedirs("Data", exist_ok=True)
    filename = f"Data/Image{index:03d}.jpg"
    image.save(filename, "JPEG")

    label_filename = f"Data/Image{index:03d}.txt"
    with open(label_filename, "w") as f:
        f.write(yolo_label)

if __name__ == "__main__":
    with Pool(cpu_count()) as pool:
        pool.map(generate_image, range(1000))
