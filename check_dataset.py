import os
from PIL import Image


train_images = os.listdir("dataset/images/train")
train_labels = os.listdir("dataset/labels/train")
val_images = os.listdir("dataset/images/val")
val_labels = os.listdir("dataset/labels/val")

print(f"Train images: {len(train_images)}, Train labels: {len(train_labels)}")
print(f"Validation images: {len(val_images)}, Validation labels: {len(val_labels)}")

def check_image_label_pairs(images, label_folder):
    for image in images:
        image_name = os.path.splitext(image)[0]
        label_name = image_name + ".txt"
        label_path = os.path.join(label_folder, label_name)
        if not os.path.exists(label_path):
            print(f"Label file for {image} not found in {label_folder}.")

def check_labels(labels, label_folder):
    for label in labels:
        label_path = os.path.join(label_folder, label)
        with open(label_path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    print(f"Invalid label format in {label_path}: {line.strip()}")
                    continue
                class_id, x_center, y_center, width, height = parts
                try:
                    if class_id != "0":
                        print(f"Unexpected class ID in {label_path}: {class_id}")
                        continue
                    class_id = int(class_id)
                    x_center = float(x_center)
                    y_center = float(y_center)
                    width = float(width)
                    height = float(height)

                except ValueError:
                    print(f"Non-numeric values in {label_path}: {line.strip()}")
                    continue

                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                    print(f"Out of bounds values in {label_path}: {line.strip()}")

def check_images(images, image_folder):
    for image in images:
        image_path = os.path.join(image_folder, image)
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify that it is an image
        except (OSError, SyntaxError) as e:
            print(f"Corrupted image file: {image_path}, Error: {e}")

check_image_label_pairs(train_images, "dataset/labels/train")
check_image_label_pairs(val_images, "dataset/labels/val")

check_labels(train_labels, "dataset/labels/train")
check_labels(val_labels, "dataset/labels/val")

check_images(train_images, "dataset/images/train")
check_images(val_images, "dataset/images/val")