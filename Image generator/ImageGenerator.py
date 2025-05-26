import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
import math
import os
from multiprocessing import Pool, cpu_count
from ImageGeneratorLib import insert_object

def generate_random_color():
    return tuple([random.randint(0, 255) for _ in range(3)])

def draw_random_square(image, width, height):
    # Generate random square size and color
    size = random.randint(50, 500)
    color = generate_random_color()

    # Ensure the square fits within the image
    x = random.randint(0, max(0, width - size))
    y = random.randint(0, max(0, height - size))

    # Create a square image
    square_img = Image.new("RGBA", (size, size), color)
    
    # Insert the square onto the image
    image_test, label = insert_object(image, square_img)

    return image_test, label

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
    #if random.random() < 0.5:
    image, yolo_label = draw_random_square(image, width, height)

    image = image.filter(ImageFilter.GaussianBlur(radius=1))

    filename = f"Data/Image{index:03d}.jpg"
    image.convert("RGB").save(filename, "JPEG")

    label_filename = f"Data/Image{index:03d}.txt"
    with open(label_filename, "w") as f:
        f.write(yolo_label)

def prepare_data_folder():
    os.makedirs("Data", exist_ok=True)
    for filename in os.listdir("Data"):
        file_path = os.path.join("Data", filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

if __name__ == "__main__":

    prepare_data_folder()

    with Pool(cpu_count()) as pool:
        pool.map(generate_image, range(1000))
