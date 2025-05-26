import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
import math
import os
from multiprocessing import Pool, cpu_count
from ImageGeneratorLib import draw_gradient_background, draw_detailed_background, draw_random_square, draw_random_ufo 


def generate_image(index):
    width, height = 640, 480
    image = draw_gradient_background(width, height)
    draw = ImageDraw.Draw(image)

    yolo_label = ""
    draw_detailed_background(draw, width, height)
    #if random.random() < 0.5:
    #    image, yolo_label = draw_random_square(image, width, height)
    if random.random() < 0.5:
        image, yolo_label = draw_random_ufo(image, width, height)        

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
