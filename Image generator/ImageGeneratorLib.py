import random
import numpy as np
from PIL import Image
import random
from PIL import ImageDraw

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

def draw_random_ufo(image, width, height):
    # Generate random UFO size
    ufo_width = random.randint(60, 300)
    ufo_height = int(ufo_width * random.uniform(0.25, 0.5))

    # Create UFO image with transparent background
    ufo_img = Image.new("RGBA", (ufo_width, ufo_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ufo_img)

    # Draw main body (ellipse)
    body_color = generate_random_color()
    draw.ellipse([0, ufo_height//4, ufo_width, ufo_height], fill=body_color)

    # Draw dome (smaller ellipse on top)
    dome_width = int(ufo_width * random.uniform(0.4, 0.7))
    dome_height = int(ufo_height * 0.5)
    dome_x0 = (ufo_width - dome_width) // 2
    dome_y0 = 0
    dome_x1 = dome_x0 + dome_width
    dome_y1 = dome_y0 + dome_height
    dome_color = generate_random_color()
    draw.ellipse([dome_x0, dome_y0, dome_x1, dome_y1], fill=dome_color)

    # Optionally add lights (small circles)
    num_lights = random.randint(3, 7)
    for i in range(num_lights):
        light_x = int((i + 1) * ufo_width / (num_lights + 1))
        light_y = int(ufo_height * 0.85)
        light_radius = max(2, ufo_width // 30)
        light_color = generate_random_color()
        draw.ellipse([
            (light_x - light_radius, light_y - light_radius),
            (light_x + light_radius, light_y + light_radius)
        ], fill=light_color)

    # Insert UFO into the image
    image_with_ufo, label = insert_object(image, ufo_img)
    return image_with_ufo, label

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

def insert_object(imgBackground, imgObject):
    """
    Inserts imgObject into imgBackground with random size, rotation, and position,
    and calculates the YOLO bounding box.
    Assumes imgObject has a black background that should be treated as transparent.

    Args:
        imgBackground: PIL Image object representing the background image.
        imgObject: PIL Image object representing the object to insert.

    Returns:
        A tuple containing:
        - A PIL Image object with the object inserted into the background.
        - A string representing the YOLO bounding box annotation, or None if the object
          is too small or outside the image bounds.
    """

    # Calculate the desired object width as a percentage of the background width
    min_width_percent = 0.10  # 10%
    max_width_percent = 0.30  # 30%
    object_width_percent = random.uniform(min_width_percent, max_width_percent)
    desired_object_width = int(imgBackground.width * object_width_percent)

    # Calculate the object height based on the desired width and the object's aspect ratio
    aspect_ratio = imgObject.width / imgObject.height
    desired_object_height = int(desired_object_width / aspect_ratio)

    # Resize the object
    imgObject = imgObject.resize((desired_object_width, desired_object_height), Image.LANCZOS)

    # Rotate the object by a random angle
    random_angle = random.uniform(0, 360)
    imgObject = imgObject.rotate(random_angle, resample=Image.BICUBIC, expand=True)

    # Ensure the background image is in RGBA mode
    if imgBackground.mode != 'RGBA':
        imgBackground = imgBackground.convert('RGBA')

    # Create a mask from the object (black background becomes transparent)
    imgObject = imgObject.convert("RGBA")
    datas = imgObject.getdata()

    new_data = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            new_data.append((0, 0, 0, 0))  # Make black pixels transparent
        else:
            new_data.append(item)
    imgObject.putdata(new_data)

    # Calculate random paste coordinates
    paste_x = random.randint(0, imgBackground.width - imgObject.width)
    paste_y = random.randint(0, imgBackground.height - imgObject.height)

    # Paste the object onto the background using the object itself as a mask
    imgBackground.paste(imgObject, (paste_x, paste_y), imgObject)

    # Calculate YOLO bounding box
    x_min = paste_x / imgBackground.width
    y_min = paste_y / imgBackground.height
    x_max = (paste_x + imgObject.width) / imgBackground.width
    y_max = (paste_y + imgObject.height) / imgBackground.height

    # Check if the object is too small or outside the image bounds
    if x_min < 0 or y_min < 0 or x_max > 1 or y_max > 1 or (x_max - x_min) < 0.001 or (y_max - y_min) < 0.001:
        return imgBackground, None  # Object is invalid

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    box_width = x_max - x_min
    box_height = y_max - y_min

    yolo_label = f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"

    return imgBackground, yolo_label