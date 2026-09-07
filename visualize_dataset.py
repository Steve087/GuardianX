from PIL import Image, ImageDraw
import os
import random

image_folder = "dataset/images/train"
label_folder = "dataset/labels/train"
image_name = "0001.jpg"  # Change this to the image you want to visualize

images = os.listdir(image_folder)
selected_image = random.sample(images, 5)

for image_name in selected_image:
    image_path = os.path.join(image_folder, image_name)
    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(label_folder, label_name)

    if not os.path.exists(label_path):
        print(f"Label file for {image_name} not found.")
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()

    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            print(f"Invalid label format in {label_path}: {line.strip()}")
            continue
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])

        # Convert YOLO format to pixel coordinates
        x_center *= img_w
        y_center *= img_h
        width *= img_w
        height *= img_h

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    img.show()