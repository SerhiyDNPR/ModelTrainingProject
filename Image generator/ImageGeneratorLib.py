from PIL import Image, ImageDraw, ImageFilter
import random
import math

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