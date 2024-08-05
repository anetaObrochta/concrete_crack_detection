import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def count_images(directory):
    return sum(len([f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]) for _, _, files in os.walk(directory))

def get_random_images(directory, n=5):
    all_images = [os.path.join(root, file)
                  for root, _, files in os.walk(directory)
                  for file in files
                  if file.endswith(('.png', '.jpg', '.jpeg'))]
    return random.sample(all_images, min(n, len(all_images))) if all_images else []

def calculate_image_stats(image_path):
    with Image.open(image_path) as img:
        img_array = np.array(img)
        return np.mean(img_array), np.std(img_array)

def verify_preprocessing(original_dir, processed_dir):
    classes = ['positive', 'negative']
    splits = ['train', 'test', 'validation']

    for cls in classes:
        orig_path = os.path.join(original_dir, cls)
        orig_count = count_images(orig_path)
        print(f"\nOriginal {cls}:")
        print(f"  Path: {orig_path}")
        print(f"  Count: {orig_count}")

        for split in splits:
            proc_path = os.path.join(processed_dir, split, cls)
            proc_count = count_images(proc_path)
            print(f"\n  Processed {split} {cls}:")
            print(f"    Path: {proc_path}")
            print(f"    Count: {proc_count}")

            orig_images = get_random_images(orig_path, 5)
            proc_images = get_random_images(proc_path, 5)

            if orig_images:
                orig_stats = [calculate_image_stats(img) for img in orig_images]
                print(f"    Original stats (mean, std): {np.mean(orig_stats, axis=0)}")
            else:
                print("    No original images found to calculate stats.")

            if proc_images:
                proc_stats = [calculate_image_stats(img) for img in proc_images]
                print(f"    Processed stats (mean, std): {np.mean(proc_stats, axis=0)}")
            else:
                print("    No processed images found to calculate stats.")

            if orig_images and proc_images:
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(Image.open(random.choice(orig_images)))
                plt.title(f"Original {cls}")
                plt.subplot(1, 2, 2)
                plt.imshow(Image.open(random.choice(proc_images)))
                plt.title(f"Processed {split}/{cls}")
                plt.show()
            else:
                print("    Not enough images to display comparison.")

# Usage
original_dir = r"C:\Users\aneta\PycharmProjects\concrete_crack_detection\data\sample"
processed_dir = r"C:\Users\aneta\PycharmProjects\concrete_crack_detection\data\processed"
verify_preprocessing(original_dir, processed_dir)