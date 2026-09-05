import os
import shutil
import random

image_source = "ExDark/People/"
label_source = "dataset/labels/people/"

files = os.listdir(image_source)
random.shuffle(files)

split = int(len(files) * 0.8)
train_files = files[:split]
val_files = files[split:]
os.makedirs("dataset/images/train", exist_ok=True)
os.makedirs("dataset/images/val", exist_ok=True)
os.makedirs("dataset/labels/train", exist_ok=True)
os.makedirs("dataset/labels/val", exist_ok=True)

print(f"Total files: {len(files)}")
print(f"Training files: {len(train_files)}")
print(f"Validation files: {len(val_files)}")

for file in train_files:
    shutil.copy(os.path.join(image_source, file), os.path.join("dataset/images/train", file))
    label_file = os.path.splitext(file)[0] + ".txt"
    shutil.copy(os.path.join(label_source, label_file), os.path.join("dataset/labels/train", label_file))

for file in val_files:
    shutil.copy(os.path.join(image_source, file), os.path.join("dataset/images/val", file))
    label_file = os.path.splitext(file)[0] + ".txt"
    shutil.copy(os.path.join(label_source, label_file), os.path.join("dataset/labels/val", label_file))

print("Dataset preparation completed.")