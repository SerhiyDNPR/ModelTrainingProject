import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
import math
import os
from multiprocessing import Pool, cpu_count
from ImageGeneratorLib import draw_gradient_background, draw_detailed_background, draw_random_square, draw_random_ufo 
from torchvision.datasets import CIFAR10
import torch
from tqdm import tqdm
import torchvision.transforms

width, height = 640, 480 #3840, 2160

def process_cifar_image(args, cifar_dataset, transform, first_image_index):
    i, idx = args
    i += first_image_index  # Adjust index to start from first_image_index
    img, _ = cifar_dataset[idx]
    img = transform(img)
    yolo_label = ""
    if random.random() < 0.5:
        img, yolo_label = draw_random_ufo(img)
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    filename = f"Data/Image{i:04d}.jpg"
    img.convert("RGB").save(filename, "JPEG")

    label_filename = f"Data/Image{i:04d}.txt"
    with open(label_filename, "w") as f:
        f.write(yolo_label)

def add_cifar_images(num_images, first_image_index):
    # Download CIFAR10 dataset
    cifar_dataset = CIFAR10(root="cifar_data", train=True, download=True)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # Convert PIL Image to Tensor
        torchvision.transforms.Resize((height, width)),
        torchvision.transforms.ToPILImage()
    ])

    indices = torch.randperm(len(cifar_dataset))[:num_images]
    args_list = [(i, idx) for i, idx in enumerate(indices)]

    with Pool(cpu_count()) as pool:
        pool.starmap(process_cifar_image, [(args, cifar_dataset, transform, first_image_index) for args in args_list])


def generate_image(index):

    image = draw_gradient_background(width, height)

    draw = ImageDraw.Draw(image)

    yolo_label = ""
    draw_detailed_background(draw, width, height)

    if random.random() < 0.5:
        image, yolo_label = draw_random_ufo(image)

    image = image.filter(ImageFilter.GaussianBlur(radius=1))

    filename = f"Data/Image{index:04d}.jpg"
    image.convert("RGB").save(filename, "JPEG")

    label_filename = f"Data/Image{index:04d}.txt"
    with open(label_filename, "w") as f:
        f.write(yolo_label)

def prepare_data_folder():
    os.makedirs("Data", exist_ok=True)
    for filename in os.listdir("Data"):
        file_path = os.path.join("Data", filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Progress for CIFAR image generation
def add_cifar_images_with_progress(num_images, first_image_index):
    cifar_dataset = CIFAR10(root="cifar_data", train=True, download=True)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((height, width)),
        torchvision.transforms.ToPILImage()
    ])
    indices = torch.randperm(len(cifar_dataset))[:num_images]
    args_list = [(i, idx) for i, idx in enumerate(indices)]

    with Pool(cpu_count()) as pool:
        list(tqdm(
            pool.starmap(
                process_cifar_image,
                [(args, cifar_dataset, transform, first_image_index) for args in args_list]
            ),
            total=num_images,
            desc="Generating CIFAR images"
        ))

if __name__ == "__main__":

    prepare_data_folder()

    num_images = 500

    # Progress for synthetic image generation
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(generate_image, range(num_images)), total=num_images, desc="Generating synthetic images"))

    add_cifar_images_with_progress(num_images, num_images)
